#!/usr/bin/env python3
"""
Validate that training data contains ONLY digit words (context-free approach).

For pure context-free training, transcriptions should only contain:
zero, one, two, three, four, five, six, seven, eight, nine

Any other words indicate context contamination.
"""

import json
from pathlib import Path
import sys
import re

# Valid digit words for context-free training
VALID_WORDS = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}

def validate_transcriptions(dataset_dir: Path):
    """Check all transcriptions for non-digit words."""

    print("=" * 80)
    print("Validating Context-Free Training Data")
    print("=" * 80)
    print(f"\nValid words: {', '.join(sorted(VALID_WORDS))}")
    print(f"Dataset: {dataset_dir}\n")

    json_files = sorted(dataset_dir.glob("*.json"))

    if not json_files:
        print(f"❌ No JSON files found in {dataset_dir}")
        return False

    print(f"Scanning {len(json_files)} files...\n")

    # Track issues
    files_with_issues = []
    all_invalid_words = set()

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        # Extract words from alignments
        words = [word for word, _, _ in data['alignments']]

        # Check each word (handle multi-word segments and punctuation)
        invalid_words = []
        for word in words:
            # Remove ALL punctuation and split into individual words
            # This handles: "one-two", "one,two", "one.two", etc.
            cleaned = re.sub(r'[^\w\s]', ' ', word)  # Replace punctuation with spaces
            word_parts = cleaned.split()

            for part in word_parts:
                # Normalize: lowercase
                normalized = part.lower()

                if normalized and normalized not in VALID_WORDS:
                    invalid_words.append(part)
                    all_invalid_words.add(normalized)

        if invalid_words:
            full_text = ' '.join(words)
            files_with_issues.append({
                'file': json_file.name,
                'text': full_text,
                'invalid': invalid_words
            })

    # Report results
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    if not files_with_issues:
        print(f"\n✅ All {len(json_files)} files are VALID (context-free)")
        print("   Only digit words found: zero, one, two, three, four, five, six, seven, eight, nine")
        return True

    print(f"\n❌ Found {len(files_with_issues)} files with CONTEXT WORDS")
    print(f"\nAll invalid words found: {', '.join(sorted(all_invalid_words))}")
    print(f"\nFiles with issues:\n")

    for i, issue in enumerate(files_with_issues, 1):
        print(f"{i}. {issue['file']}")
        print(f"   Invalid words: {', '.join(issue['invalid'])}")
        print(f"   Full text: {issue['text']}")
        print()

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("\nFor pure context-free training, you should:")
    print("  1. Remove all context words (order, number, location, is, my, etc.)")
    print("  2. Keep ONLY digit words: zero, one, two, three, four, five, six, seven, eight, nine")
    print("\nExample transformations:")
    print("  BEFORE: 'Order number is one two three four'")
    print("  AFTER:  'one two three four'")
    print("\n  BEFORE: 'My phone is five five five one two three four'")
    print("  AFTER:  'five five five one two three four'")

    return False

def main():
    dataset_dir = Path("datasets/numbers_training/processed")

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    is_valid = validate_transcriptions(dataset_dir)

    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()
