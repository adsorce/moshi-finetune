#!/usr/bin/env python3
"""
Merge LoRA adapter into base model to create a standalone fine-tuned model.
This creates a single model file that can be converted to Candle format.
"""

import argparse
import torch
from pathlib import Path
from moshi.models import loaders
from moshi.modules.lora import replace_all_linear_with_lora, replace_lora_with_linear
from safetensors.torch import load_file, save_file
import json

def merge_lora_into_base(
    checkpoint_path: Path,
    output_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
):
    """
    Merge LoRA weights into base model and save as standalone model.

    Args:
        checkpoint_path: Path to checkpoint directory containing lora.safetensors
        output_path: Where to save the merged model
        device: Device to use for merging
        dtype: Data type for model (default: bfloat16 for production compatibility)
    """

    print("=" * 80)
    print("Merging LoRA adapter into base model")
    print("=" * 80)

    # Paths
    lora_path = checkpoint_path / "lora.safetensors"
    config_path = checkpoint_path / "config.json"

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n1. Loading base STT model from HuggingFace...")
    print(f"   Model: kyutai/stt-1b-en_fr")
    print(f"   Data type: {dtype}")

    # Load base model
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo="kyutai/stt-1b-en_fr"
    )

    # Load model with LoRA configuration
    with open(config_path) as f:
        config = json.load(f)

    print(f"\n2. Initializing model with LoRA configuration...")
    print(f"   LoRA rank: {config.get('lora_rank', 128)}")
    print(f"   LoRA scaling: {config.get('lora_scaling', 2.0)}")

    # Get model
    model = checkpoint_info.get_moshi(device=device, dtype=dtype)

    print(f"\n3. Converting model to LoRA structure...")
    print(f"   Replacing Linear layers with LoRALinear layers...")

    # Convert all Linear layers to LoRALinear (required before loading LoRA weights)
    lora_rank = config.get('lora_rank', 128)
    lora_scaling = config.get('lora_scaling', 2.0)
    replace_all_linear_with_lora(model, lora_rank, lora_scaling, device=device)

    from moshi.modules.lora import LoRALinear
    lora_layer_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    print(f"   ✓ Created {lora_layer_count} LoRALinear layers")

    print(f"\n4. Loading LoRA adapter weights...")
    print(f"   From: {lora_path}")

    # Load LoRA weights
    lora_state = load_file(str(lora_path), device=device)

    print(f"   LoRA parameters: {len(lora_state)} tensors")

    # Apply LoRA weights
    result = model.load_state_dict(lora_state, strict=False)
    print(f"   ✓ Loaded LoRA weights")
    if result.unexpected_keys:
        print(f"   ⚠ Unexpected keys: {len(result.unexpected_keys)}")

    print(f"\n5. Merging LoRA into base model...")
    print(f"   This combines the adapter with the base weights...")
    print(f"   Formula: W_new = W_base + scaling * (lora_B @ lora_A)")

    # Use the built-in merge function from moshi.modules.lora
    # This function replaces all LoRALinear modules with standard Linear modules
    # containing the merged weights: W' = W + scaling * (lora_B @ lora_A)
    model.eval()

    # Count LoRA layers before merging
    lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    print(f"   Merging {lora_count} LoRA layers...")

    with torch.no_grad():
        replace_lora_with_linear(model)

    # Verify all LoRA layers were replaced
    remaining_lora = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    print(f"   ✓ Successfully merged {lora_count} LoRA layers into base weights")
    if remaining_lora > 0:
        print(f"   ⚠ Warning: {remaining_lora} LoRA layers remain unmerged!")

    print(f"\n6. Saving merged model...")
    print(f"   Output: {output_path}")

    # Get state dict
    state_dict = model.state_dict()

    # Save as safetensors
    output_file = output_path / "model.safetensors"
    save_file(state_dict, str(output_file))

    print(f"   ✓ Saved model weights: {output_file}")
    print(f"   Size: {output_file.stat().st_size / (1024**3):.2f} GB")

    # Save config
    output_config = output_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Update config to indicate LoRA has been merged
    config_data['lora_merged'] = True
    config_data['base_model'] = 'kyutai/stt-1b-en_fr'

    with open(output_config, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"   ✓ Saved config: {output_config}")

    # Also save tokenizer info if available
    print(f"\n7. Copying tokenizer information...")
    try:
        # Try to get tokenizer path from different possible attributes
        tokenizer_path = None
        for attr in ['_tokenizer_path', 'tokenizer_path', 'text_tokenizer_path']:
            if hasattr(checkpoint_info, attr):
                tokenizer_path = getattr(checkpoint_info, attr)
                break

        if tokenizer_path and Path(tokenizer_path).exists():
            import shutil
            output_tokenizer = output_path / "tokenizer.safetensors"
            shutil.copy(tokenizer_path, output_tokenizer)
            print(f"   ✓ Copied tokenizer: {output_tokenizer}")
        else:
            print(f"   ⚠ Tokenizer not found (will use HuggingFace version when loading)")
    except Exception as e:
        print(f"   ⚠ Could not copy tokenizer: {e}")
        print(f"   (Not critical - model will use HuggingFace tokenizer when loading)")

    print("\n" + "=" * 80)
    print("✓ Merge complete!")
    print("=" * 80)
    print(f"\nMerged model location: {output_path}")
    print(f"\nFiles created:")
    print(f"  - model.safetensors    (merged model weights)")
    print(f"  - config.json          (model configuration)")
    print(f"  - tokenizer.safetensors (text tokenizer)")
    print(f"\nNext steps:")
    print(f"  1. Test the merged model (optional)")
    print(f"  2. Convert to Candle format for your application")
    print(f"  3. Replace your current STT model with this one")

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./output/stt_numbers_finetune/checkpoints/checkpoint_000500/consolidated"),
        help="Path to checkpoint directory with lora.safetensors"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/merged_model"),
        help="Where to save merged model"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for merged model (default: bfloat16 for production)"
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    try:
        merge_lora_into_base(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            device=args.device,
            dtype=dtype_map[args.dtype]
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
