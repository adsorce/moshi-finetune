import json
from pathlib import Path

dataset_dir = Path("datasets/numbers_training/processed")

search_words = ['zero', 'thank', 'okay', 'yes']

print(f"Samples containing {search_words}")
print("=" * 80)

for json_file in sorted(dataset_dir.glob("*.json")):
    with open(json_file) as f:
        data = json.load(f)
    words = [word for word, _, _ in data['alignments']]
    full_text = ' '.join(words)

    found = [w for w in search_words if w in full_text.lower()]
    if found:
        print(f"{json_file.name}: {found} -> {full_text}")
