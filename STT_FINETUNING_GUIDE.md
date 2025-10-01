# STT Model Fine-Tuning Guide for Moshi

## Overview

This guide explains how to fine-tune the STT (Speech-to-Text) component of Moshi (`kyutai/stt-1b-en_fr`) using the moshi-finetune repository. The STT model is a simplified version of the full Moshi model that lacks the "depformer" component (i.e., `dep_q = 0`), which means it only performs speech-to-text transcription without generating audio responses.

## Key Differences: STT vs. Speech-to-Speech

| Aspect | Speech-to-Speech (Full Moshi) | STT Model |
|--------|-------------------------------|-----------|
| **Model Components** | Has depformer (dep_q = 8) | No depformer (dep_q = 0) |
| **Training Data** | Stereo .wav (user prompt in left channel, model response in right channel) | Mono .wav (just audio input) |
| **Output** | Text transcription + Audio response | Text transcription only |
| **Loss Computation** | text_loss + audio_loss | text_loss only |

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3090 or better recommended)
  - Example from community: RTX 3090 used ~6.5 GB peak memory
- **Python**: 3.10 or higher
- **CUDA**: Required for GPU acceleration

### Software Dependencies
Already installed via moshi-finetune:
- PyTorch with CUDA support
- moshi package (contains model definitions)
- sphn (for audio duration calculation)
- whisper_timestamped (optional, for automatic transcription)

## Required Code Patches

**IMPORTANT**: The current moshi-finetune codebase needs two patches to work with STT models. These patches disable the depformer component when `dep_q = 0`.

### Patch 1: train.py

The training loop currently always computes audio_loss, which fails for STT models. We need to make it conditional:

**Location**: `train.py` lines 267-282

**Current code**:
```python
audio_loss = compute_loss_with_mask(
    output.logits,
    codes[:, model.audio_offset : model.audio_offset + model.dep_q],
    output.mask,
    mode="audio",
    first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
)

mb_loss = text_loss + audio_loss
mb_loss.backward()

loss += mb_loss.detach()
n_batch_tokens += output.text_mask.numel() + output.mask.numel()
n_real_tokens += (
    torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
)
```

**Patched code**:
```python
# Only compute audio loss if model has depformer (dep_q > 0)
if model.dep_q > 0:
    audio_loss = compute_loss_with_mask(
        output.logits,
        codes[:, model.audio_offset : model.audio_offset + model.dep_q],
        output.mask,
        mode="audio",
        first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
    )
    mb_loss = text_loss + audio_loss
    n_batch_tokens += output.text_mask.numel() + output.mask.numel()
    n_real_tokens += (
        torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
    )
else:
    # STT model: only text loss
    mb_loss = text_loss
    n_batch_tokens += output.text_mask.numel()
    n_real_tokens += torch.sum(output.text_mask).item()

mb_loss.backward()
loss += mb_loss.detach()
```

### Patch 2: moshi/moshi/moshi/models/lm.py

The forward method currently always calls depformer, which doesn't exist in STT models:

**Location**: `moshi/moshi/moshi/models/lm.py` lines 360-371

**Current code**:
```python
logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

# map back the logits on pattern sequence to logits on original codes
logits, logits_mask = _undelay_sequence(
    self.delays[self.audio_offset:self.audio_offset + self.dep_q],
    logits, fill_value=float('NaN'))
logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float('NaN'))
text_logits_mask &= (codes[:, :1] != self.zero_token_id)
return LMOutput(logits, logits_mask, text_logits, text_logits_mask)
```

**Patched code**:
```python
# Only use depformer if model has it (dep_q > 0)
if self.dep_q > 0:
    logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

    # map back the logits on pattern sequence to logits on original codes
    logits, logits_mask = _undelay_sequence(
        self.delays[self.audio_offset:self.audio_offset + self.dep_q],
        logits, fill_value=float('NaN'))
    logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
else:
    # STT model: no depformer, no audio logits
    logits = None
    logits_mask = None

text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float('NaN'))
text_logits_mask &= (codes[:, :1] != self.zero_token_id)
return LMOutput(logits, logits_mask, text_logits, text_logits_mask)
```

## Dataset Preparation

### Dataset Format

STT training requires **mono audio files** (not stereo like speech-to-speech). The dataset structure follows the same format as the full Moshi training:

```
dataset/
├── train.jsonl                 # Table of contents
└── audio/
    ├── sample001.wav           # Mono audio file
    ├── sample001.json          # Transcription with timestamps
    ├── sample002.wav
    ├── sample002.json
    └── ...
```

### 1. Creating the .jsonl Index File

The `.jsonl` file contains metadata for each audio file:

```jsonl
{"path": "audio/sample001.wav", "duration": 24.52}
{"path": "audio/sample002.wav", "duration": 18.31}
{"path": "audio/sample003.wav", "duration": 39.39}
```

**Generate with Python**:
```python
import sphn
import json
from pathlib import Path

# Get all .wav files
wav_dir = Path("audio")
paths = [str(f) for f in wav_dir.glob("*.wav")]

# Calculate durations
durations = sphn.durations(paths)

# Write to jsonl
with open("train.jsonl", "w") as f:
    for path, duration in zip(paths, durations):
        if duration is None:
            continue
        json.dump({"path": path, "duration": duration}, f)
        f.write("\n")
```

**Alternative using shell** (requires `soxi` from sox):
```bash
cd audio/
for file in *.wav; do
    duration=$(soxi -D "$file")
    echo "{\"path\": \"audio/$file\", \"duration\": $duration}" >> ../train.jsonl
done
```

### 2. Audio File Requirements

- **Format**: WAV (16-bit PCM recommended)
- **Channels**: MONO (not stereo)
- **Sample Rate**: 16kHz or 24kHz (will be resampled by Mimi encoder)
- **Duration**: Any length (will be chunked to `duration_sec` during training)

**Convert stereo to mono**:
```bash
sox input_stereo.wav output_mono.wav channels 1
```

**Convert to proper format**:
```bash
sox input.mp3 -r 16000 -c 1 -b 16 output.wav
```

### 3. Transcription JSON Files

Each audio file needs a corresponding `.json` file with timestamped transcriptions:

**Format**:
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, how are you?"
    },
    {
      "start": 2.5,
      "end": 5.8,
      "text": "I'm doing great, thanks for asking."
    }
  ]
}
```

### 4. Generating Transcriptions

#### Option A: Using Whisper (Automatic)

The `annotate.py` script uses Whisper to generate transcriptions:

```bash
# Basic usage
python annotate.py train.jsonl

# With specific Whisper model
python annotate.py --whisper_model large-v3 train.jsonl

# With language specification
python annotate.py -l en train.jsonl

# Use local Whisper model
python annotate.py --whisper_model /path/to/whisper-large-v3 train.jsonl

# Distributed with SLURM
python annotate.py --shards 64 --partition gpu-partition train.jsonl
```

**Parameters**:
- `-l, --language`: Language code (en, fr, etc.)
- `--whisper_model`: Whisper model name or path (default: base)
- `--shards`: Number of parallel jobs for SLURM
- `--partition`: SLURM partition to use

#### Option B: Using Moshi STT (Requires Working Model)

You could use an existing Moshi STT model to generate transcriptions, though you'd need to adapt `stt_from_file_pytorch.py` to output the correct JSON format.

#### Option C: Manual Transcription

For high-quality or specialized vocabulary, manually create the JSON files with proper timestamps.

### 5. Dataset Validation

Before training, verify your dataset:

```python
import json
from pathlib import Path

# Check that all audio files have transcriptions
with open("train.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        audio_path = Path(entry["path"])
        json_path = audio_path.with_suffix(".json")

        assert audio_path.exists(), f"Missing audio: {audio_path}"
        assert json_path.exists(), f"Missing transcription: {json_path}"

        # Verify JSON format
        with open(json_path) as jf:
            data = json.load(jf)
            assert "segments" in data
            for seg in data["segments"]:
                assert all(k in seg for k in ["start", "end", "text"])

print("Dataset validation passed!")
```

## Training Configuration

Create a YAML configuration file (e.g., `stt_finetune.yaml`) based on `example/moshi_7B.yaml`:

```yaml
# Data configuration
data:
  train_data: 'path/to/train.jsonl'  # Required
  eval_data: ''                       # Optional
  shuffle: true

# Model configuration
moshi_paths:
  hf_repo_id: "kyutai/stt-1b-en_fr"  # STT model, not moshiko!

# LoRA configuration
full_finetuning: false
lora:
  enable: true
  rank: 128
  scaling: 2.
  ft_embed: true  # Important: fine-tune embeddings for STT

# Loss weighting (audio loss won't be used for STT)
first_codebook_weight_multiplier: 100.
text_padding_weight: 0.5

# Training hyperparameters
duration_sec: 30                    # Reduced from 100 for STT
batch_size: 16
max_steps: 300                      # Adjust based on dataset size
gradient_checkpointing: true

# Optimizer
optim:
  lr: 2.0e-05                       # Higher LR than speech-to-speech
  weight_decay: 0.1
  pct_start: 0.05

# Monitoring
seed: 0
log_freq: 1
eval_freq: 100
do_eval: false
do_ckpt: true
ckpt_freq: 100

# Output
save_adapters: true
run_dir: "./output/stt_finetune"

# Optional: Weights & Biases
# wandb:
#   project: "moshi-stt-finetune"
#   run_name: "stt-lora-run1"
#   key: "your-wandb-key"
#   offline: false
```

### Key Configuration Changes for STT

Compared to speech-to-speech training:

| Parameter | Speech-to-Speech | STT | Reason |
|-----------|------------------|-----|--------|
| `hf_repo_id` | `kyutai/moshiko-pytorch-bf16` | `kyutai/stt-1b-en_fr` | Different model |
| `duration_sec` | 100 | 30 | Shorter sequences for STT |
| `optim.lr` | 2e-6 | 2e-5 | Higher LR prevents loss instability |
| `lora.ft_embed` | false | true | Fine-tune embeddings for better STT |
| `max_steps` | 2000 | 300-1000 | Depends on dataset size |

## Training Process

### 1. Apply Patches

Before training, apply the two patches described above:

```bash
# Recommended: Create a branch for patches
cd /home/alex/projects/moshi-finetune
git checkout -b stt-patches

# Edit the files manually or apply patches
# (You'll need to manually edit train.py and lm.py as described above)
```

### 2. Verify Configuration

```bash
# Check that your YAML is valid
python -c "from finetune.args import TrainArgs; TrainArgs.load('stt_finetune.yaml')"
```

### 3. Start Training

```bash
# Single GPU training
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  torchrun --nproc-per-node 1 -m train stt_finetune.yaml

# Multi-GPU training (if you have multiple GPUs)
torchrun --nproc-per-node 2 --master_port $RANDOM -m train stt_finetune.yaml
```

### 4. Monitor Training

Watch the console output:
```
[train] step=1/300 | loss=2.4532 | lr=2.00e-07 | tokens/sec=1400 | mem=6.5GB
[train] step=2/300 | loss=2.1843 | lr=4.00e-07 | tokens/sec=1398 | mem=6.5GB
...
[train] step=100/300 | loss=0.3421 | lr=2.00e-05 | tokens/sec=1402 | mem=6.5GB
```

**Expected metrics**:
- **Initial loss**: 2-3 (depends on dataset)
- **Target loss**: < 0.5 (ideally close to 0 for small, specialized datasets)
- **Training speed**: ~1400 tokens/sec on RTX 3090
- **Peak memory**: ~6.5 GB on RTX 3090
- **Training time**: ~1m30s for 300 steps on 3 samples (example from community)

**Warning signs**:
- Loss jumping between 0 and >1: Learning rate too high or training instability
- Loss not decreasing: Check dataset quality, increase steps, or adjust LR
- OOM errors: Reduce `batch_size` or `duration_sec`

### 5. Checkpointing

Checkpoints are saved to `run_dir` every `ckpt_freq` steps:

```
output/stt_finetune/
├── checkpoints/
│   ├── checkpoint_000100/
│   │   └── consolidated/
│   │       ├── lora.safetensors      # LoRA weights
│   │       └── config.json            # Model config
│   ├── checkpoint_000200/
│   └── checkpoint_000300/
├── train/
│   └── metrics.jsonl                  # Training logs
└── args.yaml                          # Training config
```

## Inference / Testing

### Using the Fine-Tuned STT Model

After training, you can use your fine-tuned model for transcription:

#### With LoRA Adapters

If you saved LoRA adapters (`save_adapters: true`):

```bash
# Option 1: Using Moshi's inference script
python -m moshi.stt_from_file_pytorch \
  --audio-file input.wav \
  --lora-weight ./output/stt_finetune/checkpoints/checkpoint_000300/consolidated/lora.safetensors \
  --config-path ./output/stt_finetune/checkpoints/checkpoint_000300/consolidated/config.json

# Option 2: Custom Python script
python
```

```python
import torch
from moshi.models import loaders

# Load base model with LoRA adapter
checkpoint = loaders.CheckpointInfo.from_hf_repo(
    hf_repo="kyutai/stt-1b-en_fr"
)
model = checkpoint.get_lm_model(device="cuda")

# Load LoRA weights
lora_state = torch.load("./output/stt_finetune/checkpoints/checkpoint_000300/consolidated/lora.safetensors")
model.load_state_dict(lora_state, strict=False)

# Transcribe audio
# (You'll need to implement the transcription logic using mimi and the model)
```

#### With Full Model Weights

If you merged LoRA into the base model (`save_adapters: false`):

```bash
python -m moshi.stt_from_file_pytorch \
  --audio-file input.wav \
  --moshi-weight ./output/stt_finetune/checkpoints/checkpoint_000300/consolidated/consolidated.safetensors \
  --config-path ./output/stt_finetune/checkpoints/checkpoint_000300/consolidated/config.json
```

### Testing Model Quality

Create a test script to evaluate transcription accuracy:

```python
import json
from pathlib import Path
from moshi.models import loaders
# Add your transcription logic here

# Load test data
with open("test.jsonl") as f:
    test_samples = [json.loads(line) for line in f]

# Evaluate
for sample in test_samples:
    audio_path = sample["path"]
    ground_truth_path = Path(audio_path).with_suffix(".json")

    # Get model transcription
    predicted = transcribe_audio(audio_path, model)

    # Load ground truth
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)["segments"]

    # Compare and compute WER (Word Error Rate)
    wer = compute_wer(predicted, ground_truth)
    print(f"{audio_path}: WER = {wer:.2%}")
```

## Troubleshooting

### Common Issues

#### 1. Training Loss Jumps Around (0 → >1 → 0)

**Cause**: Learning rate too high for the dataset size or unstable gradient

**Solution**:
- Reduce learning rate: try `lr: 1.0e-05` or `lr: 5.0e-06`
- Enable gradient clipping (already enabled in train.py as `max_norm`)
- Increase batch size if memory allows
- Check for corrupted data samples

#### 2. Model Fails to Load STT Weights

**Error**: `KeyError: 'depformer'` or similar

**Cause**: Patches not applied, or wrong model specified

**Solution**:
- Verify patches are applied to both `train.py` and `lm.py`
- Check `hf_repo_id` is `kyutai/stt-1b-en_fr`, not `kyutai/moshiko-pytorch-bf16`

#### 3. Out of Memory (OOM)

**Solution** (in order of preference):
1. Reduce `batch_size` (e.g., 16 → 8 → 4)
2. Reduce `duration_sec` (e.g., 30 → 20 → 15)
3. Enable gradient checkpointing (should already be enabled)
4. Use a GPU with more VRAM

#### 4. Model Transcribes Words Incorrectly (e.g., "World" instead of "Word")

**Cause**: Insufficient training data, or model hasn't learned the correction

**Solution**:
- Increase training steps (`max_steps`)
- Add more examples of the problematic word to the dataset
- Verify transcription JSONs have correct text
- Check if overfitting: model might be too specialized

#### 5. Transcription JSON Format Issues

**Error**: Dataset loader fails or crashes

**Solution**:
- Verify JSON structure matches expected format
- Check for missing "segments" key
- Ensure timestamps don't exceed audio duration
- Validate with the dataset validation script above

## Advanced Topics

### Custom Vocabulary

If you need to teach the model specialized vocabulary (e.g., technical terms, names):

1. **Prepare targeted dataset**: Create audio samples focusing on the new vocabulary
2. **Increase training steps**: More steps help the model learn new patterns
3. **Higher learning rate**: Use `lr: 2.0e-05` or higher for faster adaptation
4. **Fine-tune embeddings**: Ensure `ft_embed: true` in config
5. **Watch for overfitting**: Test on diverse samples, not just training data

### Multi-Language Fine-Tuning

The STT model supports English and French. To fine-tune for specific accents or dialects:

1. Use language-specific Whisper model for transcription: `--whisper_model large-v3 -l fr`
2. Ensure audio samples are high-quality and representative
3. Consider separate fine-tunes for each language if mixing causes issues

### Reducing Adapter Size

If 421 MB adapters are too large:

1. Reduce LoRA rank: `rank: 64` or `rank: 32`
2. Trade-off: Lower rank = less capacity = potentially lower quality
3. Test to find the minimum rank that maintains quality

### Integration with Other Projects

If you're using this STT model in another project (like `models--kyutai--stt-1b-en_fr-candle`):

1. Export the fine-tuned model in the required format (Candle likely needs different format)
2. Check if Candle version supports LoRA adapters or needs merged weights
3. May need to convert PyTorch weights to Candle-compatible format

## Performance Expectations

### Training Performance

Based on community reports (RTX 3090, 3 training samples, 300 steps):
- **Duration**: ~1 minute 38 seconds
- **Memory**: ~6.5 GB peak VRAM
- **Speed**: ~1400 tokens/second
- **LoRA size**: ~421 MB

For larger datasets (1000+ samples):
- Scale training time proportionally
- Expect similar memory usage
- Training speed should remain consistent

### Model Quality

**Small specialized datasets** (3-50 samples):
- Can achieve near-perfect transcription on training domain
- Risk of overfitting (model may fail on out-of-domain audio)
- Best for: Correcting specific words, adapting to specific speaker

**Medium datasets** (50-1000 samples):
- Good balance of specialization and generalization
- Should improve accuracy on target domain while maintaining base capability
- Best for: Accent adaptation, domain-specific vocabulary

**Large datasets** (1000+ samples):
- Substantial improvement in target domain
- Better generalization
- May require full fine-tuning instead of LoRA for best results

## Next Steps

Once you have a working fine-tuning pipeline:

1. **Define your objective**: What specific transcription improvements do you need?
   - Fix specific word transcriptions?
   - Adapt to specific accent or speaker?
   - Add technical vocabulary?
   - Improve punctuation or formatting?

2. **Collect/prepare dataset**: Quality over quantity
   - Start small (10-50 samples) to validate pipeline
   - Gradually expand with more diverse examples
   - Ensure transcriptions are accurate

3. **Iterative improvement**:
   - Train → Test → Analyze errors → Add targeted data → Repeat
   - Track metrics (WER, specific word accuracy)
   - Document what works and what doesn't

4. **Deployment considerations**:
   - Export format (PyTorch, Candle, ONNX?)
   - Inference speed requirements
   - Memory constraints for deployment environment

## References

- [Moshi Repository](https://github.com/kyutai-labs/moshi)
- [Moshi-Finetune Repository](https://github.com/kyutai-labs/moshi-finetune)
- [STT Model on Hugging Face](https://huggingface.co/kyutai/stt-1b-en_fr)
- [DailyTalk Dataset](https://huggingface.co/datasets/kyutai/DailyTalkContiguous)
- Community source: User who successfully fine-tuned STT for "Microsoft Word" vs "Microsoft World"
