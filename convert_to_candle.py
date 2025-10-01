#!/usr/bin/env python3
"""
Convert merged PyTorch STT model to Candle format.

This script:
1. Loads the merged PyTorch model (model.safetensors)
2. Renames attention projection keys for Candle compatibility
3. Adds extra_heads (pause prediction layers) from base Candle model
4. Saves as Candle-compatible safetensors

The extra_heads are critical for VAD functionality - they predict pause
probability from the transformer output. We use the base model's extra_heads
with the fine-tuned transformer, which should naturally produce better features
for number sequences.
"""

import argparse
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import sys
import glob
import shutil
import json

def convert_to_candle(pytorch_model_path: Path, output_path: Path, to_bfloat16: bool = True):
    """
    Convert PyTorch STT model to Candle format.

    Args:
        pytorch_model_path: Path to merged PyTorch model.safetensors
        output_path: Where to save Candle-formatted model
        to_bfloat16: Convert to bfloat16 (reduces size by 50% with minimal accuracy loss)
    """
    print("=" * 80)
    print("Converting PyTorch STT Model to Candle Format")
    print("=" * 80)

    # Load PyTorch model
    print(f"\n1. Loading PyTorch model...")
    print(f"   From: {pytorch_model_path}")

    if not pytorch_model_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pytorch_model_path}")

    state_dict = load_file(str(pytorch_model_path))
    print(f"   ✓ Loaded {len(state_dict)} keys")
    print(f"   Size: {pytorch_model_path.stat().st_size / (1024**3):.2f} GB")

    candle_dict = {}
    renamed_count = 0

    # Process each key
    print(f"\n2. Converting key names for Candle compatibility...")
    print(f"   Renaming attention projection keys...")

    for key, tensor in state_dict.items():
        # Rename attention projection keys
        # PyTorch format: .in_projs.0.weight (multi-stream indexing)
        # Candle format: .in_proj_weight (single stream for STT)
        if '.in_projs.0.weight' in key:
            new_key = key.replace('.in_projs.0.weight', '.in_proj_weight')
            candle_dict[new_key] = tensor
            renamed_count += 1
        elif '.out_projs.0.weight' in key:
            new_key = key.replace('.out_projs.0.weight', '.out_proj.weight')
            candle_dict[new_key] = tensor
            renamed_count += 1
        else:
            # Copy other keys as-is
            candle_dict[key] = tensor

    print(f"   ✓ Renamed {renamed_count} attention projection keys")
    print(f"   ✓ Copied {len(state_dict) - renamed_count} keys unchanged")

    # Add extra_heads from base model
    print(f"\n3. Adding extra_heads (pause prediction layers)...")
    print(f"   These are critical for VAD functionality")
    print(f"   Loading from base Candle model cache...")

    added_heads = 0
    try:
        # Find base Candle model in HuggingFace cache
        base_candle_glob = glob.glob(
            str(Path.home() / ".cache/huggingface/hub/models--kyutai--stt-1b-en_fr-candle/snapshots/*/model.safetensors")
        )

        if base_candle_glob:
            base_candle_path = base_candle_glob[0]
            print(f"   Found: {base_candle_path}")
            base_candle = load_file(base_candle_path)

            # Copy 4 extra_heads layers
            for i in range(4):
                key = f'extra_heads.{i}.weight'
                if key in base_candle:
                    candle_dict[key] = base_candle[key]
                    shape = base_candle[key].shape
                    print(f"   ✓ Added: {key} {list(shape)}")
                    added_heads += 1

            if added_heads == 4:
                print(f"   ✓ Successfully added all 4 extra_heads layers")
            else:
                print(f"   ⚠ Warning: Only added {added_heads}/4 extra_heads")
        else:
            print(f"   ⚠ Base Candle model not found in cache")
            print(f"   Downloading base model...")

            # Download base Candle model
            from huggingface_hub import hf_hub_download
            base_path = hf_hub_download(
                repo_id="kyutai/stt-1b-en_fr-candle",
                filename="model.safetensors"
            )
            print(f"   Downloaded to: {base_path}")

            base_candle = load_file(base_path)
            for i in range(4):
                key = f'extra_heads.{i}.weight'
                if key in base_candle:
                    candle_dict[key] = base_candle[key]
                    shape = base_candle[key].shape
                    print(f"   ✓ Added: {key} {list(shape)}")
                    added_heads += 1

    except Exception as e:
        print(f"   ❌ Error adding extra_heads: {e}")
        print(f"   Model may not work correctly without extra_heads!")
        import traceback
        traceback.print_exc()

    # Convert to bfloat16 if requested
    if to_bfloat16:
        print(f"\n4. Converting to bfloat16 for production compatibility...")
        converted_count = 0
        for key, tensor in candle_dict.items():
            if tensor.dtype != torch.bfloat16:
                candle_dict[key] = tensor.to(torch.bfloat16)
                converted_count += 1
        print(f"   ✓ Converted {converted_count} tensors to bfloat16")

    # Save Candle format
    print(f"\n5. Saving Candle model...")
    print(f"   Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(candle_dict, str(output_path))

    print(f"   ✓ Saved successfully!")
    print(f"   Input keys: {len(state_dict)}")
    print(f"   Output keys: {len(candle_dict)}")
    print(f"   Size: {output_path.stat().st_size / (1024**3):.2f} GB")

    print("\n" + "=" * 80)
    print("✓ Conversion Complete!")
    print("=" * 80)

    return output_path

def copy_metadata(src_dir: Path, dst_dir: Path):
    """Copy config and tokenizer files."""
    print(f"\n5. Copying metadata files...")

    # Copy config
    src_config = src_dir / "config.json"
    dst_config = dst_dir / "config.json"
    if src_config.exists():
        shutil.copy(src_config, dst_config)
        print(f"   ✓ Copied config: {dst_config}")

        # Update config to indicate Candle format
        with open(dst_config, 'r') as f:
            config = json.load(f)
        config['format'] = 'candle'
        config['converted_from'] = str(src_dir / "model.safetensors")
        with open(dst_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"   ✓ Updated config with format info")

    # Copy tokenizer if exists
    src_tokenizer = src_dir / "tokenizer.safetensors"
    dst_tokenizer = dst_dir / "tokenizer.safetensors"
    if src_tokenizer.exists():
        shutil.copy(src_tokenizer, dst_tokenizer)
        print(f"   ✓ Copied tokenizer: {dst_tokenizer}")
    else:
        print(f"   ⚠ No tokenizer file (will use HuggingFace version)")

def verify_conversion(candle_path: Path):
    """Verify the converted Candle model."""
    print(f"\n6. Verifying conversion...")

    from safetensors import safe_open

    with safe_open(str(candle_path), framework='pt') as f:
        keys = list(f.keys())

    # Check for proper formatting
    extra_heads = [k for k in keys if 'extra_heads' in k]
    in_proj = [k for k in keys if 'in_proj_weight' in k]
    old_format = [k for k in keys if 'in_projs.0' in k or 'out_projs.0' in k]

    print(f"   Total keys: {len(keys)}")
    print(f"   Extra heads: {len(extra_heads)} (expected: 4)")
    print(f"   New format (in_proj_weight): {len(in_proj)} layers")
    print(f"   Old format (in_projs.0): {len(old_format)} (should be 0)")

    success = True
    if len(extra_heads) != 4:
        print(f"   ⚠ Warning: Expected 4 extra_heads, found {len(extra_heads)}")
        success = False
    if len(in_proj) == 0:
        print(f"   ⚠ Warning: No in_proj_weight keys found")
        success = False
    if len(old_format) > 0:
        print(f"   ⚠ Warning: Old format keys still present")
        success = False

    if success:
        print(f"\n   ✓ Verification successful! Model ready for deployment.")
    else:
        print(f"\n   ⚠ Verification warnings - please review")

    return success

def main():
    parser = argparse.ArgumentParser(
        description="Convert merged PyTorch STT model to Candle format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./output/merged_model/model.safetensors"),
        help="Path to merged PyTorch model.safetensors"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/candle_model/model.safetensors"),
        help="Where to save Candle model"
    )
    parser.add_argument(
        "--to-bfloat16",
        action="store_true",
        default=True,
        help="Convert to bfloat16 format (default: True for production compatibility)"
    )

    args = parser.parse_args()

    try:
        # Convert model
        output_path = convert_to_candle(args.input, args.output, args.to_bfloat16)

        # Copy metadata
        copy_metadata(args.input.parent, args.output.parent)

        # Verify
        verify_conversion(output_path)

        print(f"\n" + "=" * 80)
        print("Next Steps:")
        print("=" * 80)
        print(f"\nCandle model location: {args.output.parent}")
        print(f"\nFiles created:")
        print(f"  - model.safetensors    (Candle-formatted fine-tuned model)")
        print(f"  - config.json          (model configuration)")
        print(f"\nDeployment:")
        print(f"  1. Test the Candle model in your application")
        print(f"  2. Verify transcription outputs word-form numbers")
        print(f"  3. Verify VAD behavior (no premature cut-offs)")
        print(f"  4. Replace your production STT model with this one")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
