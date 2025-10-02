#!/bin/bash
# Recording and Processing Script for STT Training Dataset (macOS version)
# This script mirrors record_and_process.sh but targets macOS audio tooling.

set -e

# Configuration
DATASET_DIR="./datasets/numbers_training"
RAW_DIR="${DATASET_DIR}/raw"
PROCESSED_DIR="${DATASET_DIR}/processed"
SCRIPT_FILE="./RECORDING_SCRIPT.md"
SAMPLE_RATE=16000
RECORD_DURATION=10  # seconds, can be adjusted
INPUT_SAMPLE_RATE=44100  # Set to match macOS Audio MIDI Setup input sample rate
INPUT_CHANNELS=2
INPUT_BIT_DEPTH=16

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "${RAW_DIR}"
mkdir -p "${PROCESSED_DIR}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  STT Training Data Recording Tool${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check dependencies
check_dependencies() {
    local missing=0

    if ! command -v sox >/dev/null 2>&1; then
        echo -e "${RED}Error: sox is not installed${NC}"
        echo "Install with: brew install sox"
        missing=1
    fi

    if ! command -v rec >/dev/null 2>&1; then
        echo -e "${RED}Error: The 'rec' command from SoX is unavailable${NC}"
        echo "Ensure SoX binaries are on PATH (brew install sox)"
        missing=1
    fi

    if ! command -v play >/dev/null 2>&1; then
        echo -e "${RED}Error: The 'play' command from SoX is unavailable${NC}"
        echo "Ensure SoX binaries are on PATH (brew install sox)"
        missing=1
    fi

    if [ $missing -eq 1 ]; then
        exit 1
    fi
}

# Function to display the script for a sample number
show_script() {
    local sample_num=$1
    local script_content

    echo -e "${YELLOW}Sample ${sample_num} Script:${NC}"
    # Use literal match to avoid regex escaping issues in markdown headers
    if script_content=$(grep -F -A 1 "**Sample ${sample_num}:**" "${SCRIPT_FILE}" 2>/dev/null); then
        printf '%s\n' "${script_content}" | sed -e '/^--$/d' -e 's/^/    /'
    else
        echo "    Check RECORDING_SCRIPT.md for sample ${sample_num}"
    fi
    echo ""
}

# Function to clear the terminal between recordings for readability
clear_screen_for_sample() {
    if [ -t 1 ]; then
        printf '\033[2J\033[H'
    fi
}

# Function to record a single sample
record_sample() {
    local sample_num=$1
    local padded_num=$(printf "%03d" $sample_num)
    local raw_file="${RAW_DIR}/sample_${padded_num}_raw.wav"
    local processed_file="${PROCESSED_DIR}/sample_${padded_num}.wav"

    # Check if already processed
    if [ -f "${processed_file}" ]; then
        echo -e "${YELLOW}Sample ${sample_num} already exists. ${NC}"
        read -p "Overwrite? (y/n): " overwrite
        if [ "$overwrite" != "y" ]; then
            return
        fi
    fi

    clear_screen_for_sample

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Recording Sample ${sample_num}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    echo -e "${GREEN}Ready to record Sample ${sample_num}${NC}"
    echo "Press ENTER to start recording..."
    read dummy </dev/tty

    echo -e "${RED}Now recording... Press ENTER to stop early${NC}"
    echo "(Recording auto-stops after ${RECORD_DURATION} seconds)"
    echo ""

    # Start recording in the background so we can stop immediately when the user is done.
    rec --clobber -r ${INPUT_SAMPLE_RATE} -c ${INPUT_CHANNELS} -b ${INPUT_BIT_DEPTH} "${raw_file}" trim 0 ${RECORD_DURATION} >/dev/null 2>&1 &
    local rec_pid=$!

    # Show the script beneath the recording status so it's easy to read while recording.
    show_script $sample_num

    # Stop the recorder if the script receives Ctrl+C.
    trap 'kill -INT ${rec_pid} 2>/dev/null || true' INT

    # Wait for ENTER (or timeout) without blocking the recorder from running.
    if read -r -s -n 1 -t ${RECORD_DURATION} dummy </dev/tty; then
        kill -INT ${rec_pid} 2>/dev/null || true
    fi

    # Ensure recorder fully stops before proceeding.
    wait ${rec_pid} 2>/dev/null || true

    # Reset trap
    trap - INT

    echo -e "${GREEN}Recording stopped!${NC}"

    # Process: Convert to mono, 16kHz, 16-bit
    echo "Processing audio..."
    sox "${raw_file}" -r ${SAMPLE_RATE} -c 1 -b 16 "${processed_file}"

    # Play back for verification
    echo -e "${BLUE}Playing back recording...${NC}"
    play "${processed_file}" 2>/dev/null

    # Ask if satisfied
    read -p "Keep this recording? (y/n/r for re-record): " response </dev/tty
    case $response in
        y|Y)
            echo -e "${GREEN}Sample ${sample_num} saved!${NC}"
            ;;
        r|R)
            echo "Re-recording..."
            rm -f "${raw_file}" "${processed_file}"
            record_sample $sample_num
            ;;
        *)
            echo "Discarding..."
            rm -f "${raw_file}" "${processed_file}"
            ;;
    esac

    echo ""
}

# Function to record multiple samples
record_batch() {
    local start_num=$1
    local end_num=$2

    for i in $(seq $start_num $end_num); do
        record_sample $i

        if [ $i -lt $end_num ]; then
            echo -e "${BLUE}Next sample coming up...${NC}"
            echo "Press ENTER to continue (or Ctrl+C to stop)..."
            read
        fi
    done
}

# Function to create dataset index (jsonl)
create_index() {
    local jsonl_file="${DATASET_DIR}/train.jsonl"

    echo "Creating dataset index..."

    # Ensure dataset directory exists
    mkdir -p "${DATASET_DIR}"

    # Remove existing index
    rm -f "${jsonl_file}"

    # Check if we have any wav files
    local wav_count=$(ls -1 "${PROCESSED_DIR}"/*.wav 2>/dev/null | wc -l)
    if [ "$wav_count" -eq 0 ]; then
        echo -e "${RED}No WAV files found in ${PROCESSED_DIR}${NC}"
        echo "Record some samples first (option 1, 2, or 3)"
        return
    fi

    # Generate new index
    for wav_file in "${PROCESSED_DIR}"/*.wav; do
        if [ -f "${wav_file}" ]; then
            # Get duration using soxi
            duration=$(soxi -D "${wav_file}" 2>/dev/null || echo "0")

            # Get relative path
            rel_path="processed/$(basename ${wav_file})"

            # Write to jsonl
            echo "{\"path\": \"${rel_path}\", \"duration\": ${duration}}" >> "${jsonl_file}"
        fi
    done

    local count=$(wc -l < "${jsonl_file}" 2>/dev/null || echo "0")
    echo -e "${GREEN}Created index with ${count} samples${NC}"
}

# Function to run Whisper transcription
run_transcription() {
    local jsonl_file="${DATASET_DIR}/train.jsonl"

    if [ ! -f "${jsonl_file}" ]; then
        echo -e "${RED}Error: train.jsonl not found. Run option 4 first.${NC}"
        return
    fi

    echo -e "${BLUE}Running Whisper transcription...${NC}"
    echo "This may take a few minutes..."

    cd "${DATASET_DIR}"
    python ../../annotate.py -l --whisper_model large-v3-turbo train.jsonl
    cd - > /dev/null

    echo -e "${GREEN}Transcription complete!${NC}"
    echo -e "${YELLOW}IMPORTANT: You must now manually edit the .json files to convert digits to words!${NC}"
    echo "Location: ${PROCESSED_DIR}/*.json"
}

# Function to show status
show_status() {
    local processed_count=$(ls -1 "${PROCESSED_DIR}"/*.wav 2>/dev/null | wc -l)
    local json_count=$(ls -1 "${PROCESSED_DIR}"/*.json 2>/dev/null | wc -l)

    echo -e "${BLUE}Dataset Status:${NC}"
    echo "  Processed audio files: ${processed_count}"
    echo "  Transcription files:   ${json_count}"
    echo ""

    if [ $processed_count -gt 0 ]; then
        echo "Recent samples:"
        ls -1t "${PROCESSED_DIR}"/*.wav | head -5 | while read f; do
            echo "  - $(basename $f)"
        done
    fi
}

# Main menu
main_menu() {
    while true; do
        echo ""
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}  Main Menu${NC}"
        echo -e "${BLUE}========================================${NC}"
        show_status
        echo ""
        echo "1. Record single sample"
        echo "2. Record batch (multiple samples)"
        echo "3. Record all 200 samples (interactive)"
        echo "4. Create dataset index (train.jsonl)"
        echo "5. Run Whisper transcription"
        echo "6. Show dataset status"
        echo "7. Exit"
        echo ""
        read -p "Select option: " choice

        case $choice in
            1)
                read -p "Enter sample number (1-200): " num
                if [ $num -ge 1 ] && [ $num -le 200 ]; then
                    record_sample $num
                else
                    echo -e "${RED}Invalid sample number (must be 1-200)${NC}"
                fi
                ;;
            2)
                read -p "Enter start number: " start
                read -p "Enter end number: " end
                if [ $start -ge 1 ] && [ $end -le 200 ] && [ $start -le $end ]; then
                    record_batch $start $end
                else
                    echo -e "${RED}Invalid range (must be 1-200)${NC}"
                fi
                ;;
            3)
                record_batch 1 200
                ;;
            4)
                create_index
                ;;
            5)
                run_transcription
                ;;
            6)
                show_status
                ;;
            7)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                ;;
        esac
    done
}

# Run checks and start
check_dependencies
main_menu
