# Complete Recording Workflow Guide

## Overview

This workflow provides an interactive system to record, process, and prepare audio samples for STT training. Everything is automated except for the actual speaking!

## Quick Start

```bash
cd /home/alex/projects/moshi-finetune
./record_and_process.sh
```

## What You Need

### Software (check if installed)
```bash
# Check if sox is installed
sox --version

# Check if arecord is installed
arecord --version

# If missing, install:
sudo pacman -S sox alsa-utils
```

### Hardware
- A microphone (built-in laptop mic, USB mic, or headset mic all work)
- Quiet environment

## Complete Workflow

### Step 1: Start the Recording Tool

```bash
./record_and_process.sh
```

You'll see a menu with these options:
```
Main Menu
========================================
Dataset Status:
  Processed audio files: 0
  Transcription files:   0

1. Record single sample
2. Record batch (multiple samples)
3. Record all 50 samples (interactive)
4. Create dataset index (train.jsonl)
5. Run Whisper transcription
6. Show dataset status
7. Exit
```

### Step 2: Record Your Samples

**Recommended approach: Option 2 (Record batch)**

Choose option 2 and record in batches:
- Batch 1-10: Warm up, get comfortable
- Batch 11-20: Continue with variety
- Etc.

**How it works:**
1. Script shows you the text to read (from RECORDING_SCRIPT.md)
2. Press ENTER when ready
3. Script records for 10 seconds (you can speak shorter)
4. Recording plays back automatically
5. Choose: Keep (y), Discard (n), or Re-record (r)
6. Continues to next sample

**Tips during recording:**
- Speak naturally, don't over-enunciate
- Wait 1 second before starting to speak
- Finish speaking before the 10 seconds ends
- Say "zero" not "oh" for the digit 0
- You can pause between batches anytime (Ctrl+C)

### Step 3: Create Dataset Index

After recording all samples (or a batch):

Choose option 4: **Create dataset index (train.jsonl)**

This automatically:
- Scans all recorded WAV files
- Calculates durations using `soxi`
- Creates `datasets/numbers_training/train.jsonl`

### Step 4: Run Whisper Transcription

Choose option 5: **Run Whisper transcription**

This automatically:
- Runs Whisper on all audio files
- Creates `.json` files with timestamps
- Takes 2-5 minutes for 50 samples

**Note:** Whisper will transcribe numbers as DIGITS (1234567), not words. We'll fix this next.

### Step 5: Convert Digits to Words

Exit the recording tool (option 7) and run:

```bash
# First, test it (dry run)
python convert_transcriptions.py --dry-run

# If it looks good, run for real
python convert_transcriptions.py
```

This script:
- Reads all `.json` transcription files
- Converts digit sequences to word format
  - "1234567" → "one two three four five six seven"
  - "310-555-1234" → "three one zero five five five one two three four"
- Saves the updated files

**Example output:**
```
Found 50 transcription files

sample_001.json:
  'order number is 1296803' -> 'order number is one two nine six eight zero three'
  ✓ Saved

sample_002.json:
  'order number 3456789' -> 'order number three four five six seven eight nine'
  ✓ Saved

...

Processed: 50 files
✓ Conversion complete!
```

### Step 6: Manual Review (Important!)

Open a few `.json` files to verify:

```bash
kate datasets/numbers_training/processed/sample_001.json
```

Check that:
- Transcription is accurate
- Numbers are in word format
- Timestamps look reasonable
- No weird artifacts from Whisper

**Example JSON structure:**
```json
{
  "alignments": [
    ["The", [0.0, 0.2], "SPEAKER_MAIN"],
    ["order", [0.2, 0.5], "SPEAKER_MAIN"],
    ["number", [0.5, 0.9], "SPEAKER_MAIN"],
    ["is", [0.9, 1.1], "SPEAKER_MAIN"],
    ["one", [1.1, 1.3], "SPEAKER_MAIN"],
    ["two", [1.3, 1.5], "SPEAKER_MAIN"],
    ...
  ]
}
```

### Step 7: Verify Dataset Structure

Your directory should look like this:

```
datasets/numbers_training/
├── train.jsonl                      # Dataset index
├── raw/                             # Original recordings (can delete)
│   ├── sample_001_raw.wav
│   ├── sample_002_raw.wav
│   └── ...
└── processed/                       # Training data
    ├── sample_001.wav               # Mono, 16kHz WAV
    ├── sample_001.json              # Transcription with timestamps
    ├── sample_002.wav
    ├── sample_002.json
    └── ...
```

Verify:
```bash
# Check WAV properties
soxi datasets/numbers_training/processed/sample_001.wav
# Should show: 16000 Hz, Mono, 16-bit

# Check dataset index
cat datasets/numbers_training/train.jsonl | head -3
```

### Step 8: Ready to Train!

You're now ready to create the training config and start training. See the next section.

## Directory Structure Created

```
moshi-finetune/
├── record_and_process.sh            # Main recording tool
├── convert_transcriptions.py        # Digit→word converter
├── RECORDING_SCRIPT.md             # Scripts for all 50 samples
├── RECORDING_WORKFLOW.md           # This guide
└── datasets/
    └── numbers_training/
        ├── train.jsonl              # Dataset index (for training)
        ├── raw/                     # Original recordings (optional)
        │   └── sample_*.wav
        └── processed/               # Processed training data
            ├── sample_001.wav       # Mono 16kHz audio
            ├── sample_001.json      # Transcription
            ├── sample_002.wav
            ├── sample_002.json
            └── ...
```

## File Formats

### Audio Files (sample_001.wav)
- **Format:** WAV (PCM signed 16-bit little endian)
- **Channels:** Mono
- **Sample Rate:** 16000 Hz (16 kHz)
- **Duration:** Typically 3-10 seconds

### Dataset Index (train.jsonl)
```json
{"path": "processed/sample_001.wav", "duration": 4.523}
{"path": "processed/sample_002.wav", "duration": 5.891}
...
```

### Transcription Files (sample_001.json)
```json
{
  "alignments": [
    ["word1", [start_time, end_time], "SPEAKER_MAIN"],
    ["word2", [start_time, end_time], "SPEAKER_MAIN"],
    ...
  ]
}
```

## Troubleshooting

### "arecord: command not found"
```bash
sudo pacman -S alsa-utils
```

### "sox: command not found"
```bash
sudo pacman -S sox
```

### Recording sounds bad / wrong device
List available recording devices:
```bash
arecord -l
```

Modify `record_and_process.sh` line with arecord to specify device:
```bash
arecord -D hw:1,0 -f S16_LE -r 48000 -c 2 -d ${RECORD_DURATION} "${raw_file}"
```

### No sound during playback
Check audio output:
```bash
aplay -l
```

Or skip playback verification - just trust your recordings.

### Whisper taking too long
The script uses `whisper-large-v3-turbo` by default (fast).
If you have it installed, it should be quick (~2-5 minutes for 50 samples).

### Python script errors
Make sure you're in the correct directory:
```bash
cd /home/alex/projects/moshi-finetune
python convert_transcriptions.py
```

### Need to re-record a specific sample
```bash
./record_and_process.sh
# Choose option 1 (single sample)
# Enter the sample number you want to redo
```

## Tips for Best Results

### Recording Tips
1. **Consistency:** Try to record all samples in one or two sessions
2. **Environment:** Same room, same mic, same time of day
3. **Variety:** Vary your pace and tone naturally
4. **Mistakes:** If you mess up, just re-record (option 'r')
5. **Breaks:** Take breaks every 10-15 samples to keep your voice fresh

### Quality Checks
- Listen to a few recordings to ensure audio quality
- Check that numbers are clearly spoken
- Verify transcriptions are accurate before training
- Make sure no recordings are cut off or too short

### Efficiency
- Record in batches of 10-15 samples
- Don't worry about perfection - natural variation is good
- You can always record more samples later to improve the model

## What Happens Behind the Scenes

1. **Recording:** Uses `arecord` to capture audio (stereo, 48kHz)
2. **Processing:** Uses `sox` to convert to mono 16kHz
3. **Playback:** Uses `aplay` for verification
4. **Index Creation:** Uses `soxi` to get durations
5. **Transcription:** Uses Whisper via `annotate.py`
6. **Conversion:** Python script converts digits to words

## Next Steps After Recording

Once you have your dataset ready:

1. **Create training config** (see next guide)
2. **Start training:**
   ```bash
   torchrun --nproc-per-node 1 -m train stt_numbers.yaml
   ```
3. **Monitor training** (loss should decrease)
4. **Test the model** (verify VAD behavior)
5. **Convert to Candle** (for your production use)

---

**Estimated Time:**
- Recording 50 samples: 30-45 minutes (with re-records)
- Processing & transcription: 5-10 minutes
- Manual review: 10-15 minutes
- **Total: ~1 hour**
