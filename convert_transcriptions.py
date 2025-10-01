#!/usr/bin/env python3
"""
Convert Whisper transcriptions to context-free word format.
This script processes all .json files in the processed directory and:
1. Converts digit numbers to their word equivalents (1 -> one)
2. Removes all context words (order, number, location, etc.)
3. Removes all punctuation
4. Leaves ONLY digit words: zero, one, two, three, four, five, six, seven, eight, nine
"""

import json
import re
from pathlib import Path
import argparse

# Digit to word mapping
DIGIT_TO_WORD = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine'
}

# Valid digit words for context-free training
VALID_DIGIT_WORDS = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}

def convert_to_context_free(text: str) -> str:
    """
    Convert text to context-free format with ONLY digit words.

    Steps:
    1. Convert any digits (1, 2, 3) to word form (one, two, three)
    2. Remove ALL punctuation
    3. Keep ONLY digit words, remove all other words
    4. Return space-separated digit words

    Examples:
        "order 1234567" -> "one two three four five six seven"
        "My phone is 555-1234" -> "five five five one two three four"
        "Location code 444-555" -> "four four four five five five"
        "one two three-four five" -> "one two three four five"
    """
    # Step 1: Convert any digits to word form (with spaces between)
    result = []
    i = 0
    while i < len(text):
        if text[i].isdigit():
            # Convert digit to word and add space after
            result.append(DIGIT_TO_WORD[text[i]] + ' ')
            i += 1
        else:
            result.append(text[i])
            i += 1

    text = ''.join(result)

    # Step 2: Remove ALL punctuation (replace with spaces)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Step 3: Split into words and keep ONLY digit words
    words = text.split()
    digit_words = [w.lower() for w in words if w.lower() in VALID_DIGIT_WORDS]

    # Step 4: Join with spaces
    return ' '.join(digit_words)

def process_whisper_json(json_path: Path, dry_run: bool = False) -> tuple[bool, str]:
    """
    Process a single Whisper JSON file and convert to context-free format.

    Args:
        json_path: Path to the JSON file
        dry_run: If True, don't save changes

    Returns:
        (success: bool, message: str)
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Handle the Whisper annotate.py format
        # Format: {"alignments": [["word", [start, end], "SPEAKER_MAIN"], ...]}
        if 'alignments' not in data:
            return False, f"Unexpected format in {json_path.name}"

        conversions = []
        for alignment in data['alignments']:
            if len(alignment) >= 3:
                original_word = alignment[0]
                converted_word = convert_to_context_free(original_word)

                if original_word != converted_word:
                    conversions.append(f"  '{original_word}' -> '{converted_word}'")

                alignment[0] = converted_word

        # Remove empty alignments (where all context was stripped)
        data['alignments'] = [a for a in data['alignments'] if a[0]]

        if conversions:
            msg = f"{json_path.name}:\n" + "\n".join(conversions)

            if not dry_run:
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                msg += f"\n  ✓ Saved"
            else:
                msg += f"\n  (dry run - not saved)"

            return True, msg
        else:
            return True, f"{json_path.name}: No changes needed"

    except Exception as e:
        return False, f"Error processing {json_path.name}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(
        description="Convert Whisper transcriptions to context-free format (ONLY digit words)"
    )
    parser.add_argument(
        'directory',
        type=Path,
        nargs='?',
        default=Path('./datasets/numbers_training/processed'),
        help='Directory containing .json transcription files (default: ./datasets/numbers_training/processed)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without actually modifying files'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests on sample text'
    )

    args = parser.parse_args()

    # Run tests if requested
    if args.test:
        print("Running context-free conversion tests...")
        print("=" * 80)
        test_cases = [
            ("order 1234567", "one two three four five six seven"),
            ("My phone is 555-1234", "five five five one two three four"),
            ("Location code 444-555", "four four four five five five"),
            ("one two three-four five", "one two three four five"),
            ("The number is 999888", "nine nine nine eight eight eight"),
            ("no numbers here", ""),  # Empty - no digit words
            ("Call me at 310-555-1234", "three one zero five five five one two three four"),
        ]

        all_passed = True
        for input_text, expected in test_cases:
            result = convert_to_context_free(input_text)
            passed = result == expected
            status = "✓" if passed else "✗"
            print(f"{status} '{input_text}'")
            print(f"   Result:   '{result}'")
            if not passed:
                print(f"   Expected: '{expected}'")
                all_passed = False
            print()

        print("=" * 80)
        print("All tests passed!" if all_passed else "Some tests failed!")
        return

    # Process directory
    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        print("Have you run the recording script yet?")
        return

    json_files = list(args.directory.glob("*.json"))

    if not json_files:
        print(f"No .json files found in {args.directory}")
        print("Have you run Whisper transcription yet? (Option 5 in recording script)")
        return

    print(f"Found {len(json_files)} transcription files")
    print("MODE: Context-free (removing all non-digit words)\n")

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified\n")

    success_count = 0
    error_count = 0

    for json_file in sorted(json_files):
        success, message = process_whisper_json(json_file, args.dry_run)
        print(message)
        print()

        if success:
            success_count += 1
        else:
            error_count += 1

    print("=" * 50)
    print(f"Processed: {success_count} files")
    if error_count > 0:
        print(f"Errors: {error_count} files")

    if not args.dry_run:
        print("\n✓ Conversion complete!")
        print("\nNext steps:")
        print("1. Manually review the .json files to verify conversions")
        print("2. Create the training configuration file")
        print("3. Start training!")
    else:
        print("\nRun without --dry-run to actually modify files")

if __name__ == '__main__':
    main()
