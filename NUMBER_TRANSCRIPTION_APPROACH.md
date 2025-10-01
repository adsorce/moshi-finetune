# Approach: Fixing VAD Cut-off During Number Speech

## Problem Statement

The Moshi STT model uses **extra heads** to predict pauses, which determines when to end the user's turn (VAD - Voice Activity Detection). When users speak digit numbers (1, 2, 3...), the pause prediction spikes above the threshold and cuts off the user mid-sentence. However, when the model transcribes numerals as words (one, two, three), the pause prediction doesn't spike and the user can continue speaking.

**Key Observation**: The transcription format (digits vs. words) correlates with pause prediction behavior, suggesting the model has learned different pause patterns for different transcription styles.

## Architecture: Extra Heads for Pause Prediction

Based on code analysis:

### Model Components

1. **Main Transformer**: Processes audio tokens and produces text transcription
2. **Extra Heads** (`extra_heads`): Additional linear layers that predict auxiliary information
   - Defined in `lm.py:218-220`
   - Configuration: `extra_heads_num_heads` (default: 0) and `extra_heads_dim` (default: 6)
   - Each head is a `Linear(dim, extra_heads_dim, bias=False)` layer

3. **Inference Usage** (`batched_asr.py:196`):
   ```python
   text_tokens, extra_heads_list = self.lm_gen.step_with_extra_heads(audio_tokens)
   extra_heads_stacked = torch.stack(extra_heads_list, dim=0)
   extra_heads_stacked = extra_heads_stacked[:, :, 0, 0]
   ```
   - The extra heads output is used for pause/turn-end detection
   - Softmax is applied to get probabilities (`lm.py:807`)

### How It Works

The extra heads predict pause likelihood from the transformer's internal representation:
- **Input**: Transformer output embeddings
- **Output**: Probability distribution over pause states (dimension 6)
- **Usage**: Client code (Candle/Rust) monitors these probabilities to detect turn end

## Available Approaches

### Approach 1: Train to Always Transcribe Numbers as Words ⭐ **RECOMMENDED**

**Strategy**: Fine-tune the STT model to consistently output word-form numerals (one, two, three) instead of digit-form (1, 2, 3).

#### Why This Works
- The model already handles word-form numerals correctly (no pause spike)
- Only requires modifying text transcription behavior
- Doesn't require understanding or modifying pause prediction mechanism
- Leverages existing model capability

#### Dataset Preparation

1. **Audio Samples**: Record/collect audio of people saying numbers
   - Phone numbers: "My number is three one zero five five five..."
   - Zip codes: "The code is nine zero two one zero"
   - Addresses: "I live at one two three four Main Street"
   - Counting: "One, two, three, four, five..."
   - Mixed contexts: "I need two pizzas for three people at seven PM"

2. **Transcription Format**: Use Whisper or manual transcription with **word-form numerals**

   Example JSON:
   ```json
   {
     "segments": [
       {
         "start": 0.0,
         "end": 3.5,
         "text": "My phone number is three one zero five five five one two three four."
       },
       {
         "start": 4.0,
         "end": 6.8,
         "text": "The zip code is nine zero two one zero."
       }
     ]
   }
   ```

3. **Dataset Size Estimate**:
   - **Minimum**: 50-100 samples (3-10 seconds each) focusing on number sequences
   - **Recommended**: 200-500 samples with diverse contexts
   - **Ideal**: 1000+ samples covering various accents, speeds, and contexts

#### Implementation Steps

1. **Create Training Data**:
   ```bash
   # Record or collect audio files with number speech
   # Ensure they are MONO .wav files (16kHz recommended)

   # Option A: Auto-transcribe with Whisper, then manually convert digits to words
   python annotate.py train.jsonl
   # Then manually edit .json files: change "310" to "three one zero"

   # Option B: Manually transcribe directly with word-form numerals
   ```

2. **Transcription Conversion Script** (helper):
   ```python
   import json
   import re
   from pathlib import Path

   DIGIT_TO_WORD = {
       '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
       '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
   }

   def convert_digits_to_words(text):
       """Convert isolated digits to word form."""
       def replace_digit(match):
           return DIGIT_TO_WORD[match.group(0)]

       # Match individual digits (not part of larger numbers)
       # This regex handles "1 2 3" but not years like "2024"
       pattern = r'\b(\d)\b'
       return re.sub(pattern, replace_digit, text)

   # Process all transcription files
   for json_file in Path("audio/").glob("*.json"):
       with open(json_file) as f:
           data = json.load(f)

       for segment in data["segments"]:
           segment["text"] = convert_digits_to_words(segment["text"])

       with open(json_file, 'w') as f:
           json.dump(data, f, indent=2)

   print("Converted digits to words in all transcriptions")
   ```

3. **Training Configuration** (`stt_numbers.yaml`):
   ```yaml
   data:
     train_data: 'datasets/numbers_train.jsonl'
     eval_data: ''
     shuffle: true

   moshi_paths:
     hf_repo_id: "kyutai/stt-1b-en_fr"

   full_finetuning: false
   lora:
     enable: true
     rank: 128
     scaling: 2.
     ft_embed: true  # Important for learning new transcription patterns

   first_codebook_weight_multiplier: 100.
   text_padding_weight: 0.5

   duration_sec: 30
   batch_size: 16
   max_steps: 500  # Adjust based on dataset size
   gradient_checkpointing: true

   optim:
     lr: 2.0e-05
     weight_decay: 0.1
     pct_start: 0.05

   seed: 0
   log_freq: 10
   eval_freq: 100
   do_eval: false
   do_ckpt: true
   ckpt_freq: 100

   save_adapters: true
   run_dir: "./output/stt_numbers_finetune"
   ```

4. **Training**:
   ```bash
   # Apply patches first (already done)

   # Train
   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
     torchrun --nproc-per-node 1 -m train stt_numbers.yaml
   ```

5. **Testing**:
   - Create test set with number sequences
   - Verify transcriptions use word-form
   - Test that pause prediction doesn't spike during number sequences

#### Pros & Cons

✅ **Pros**:
- Simpler implementation (text-only modification)
- No need to understand pause prediction internals
- Leverages existing working behavior
- Focused dataset requirement
- Can be done with moderate dataset size (50-500 samples)

❌ **Cons**:
- Output format changes (may affect downstream systems expecting digits)
- Need to convert numbers back to digits if required by application
- May affect non-sequence numbers (e.g., "I need 2 pizzas" → "I need two pizzas")

---

### Approach 2: Train Extra Heads to Reduce Pause Prediction on Numbers

**Strategy**: Fine-tune the extra heads (pause prediction) to not spike when digits are spoken, while keeping transcription as-is.

#### Why This Is Challenging

1. **No Direct Loss**: The current training code doesn't compute loss for extra heads
   - Extra heads are defined but not trained in standard pipeline
   - Would need to add custom loss computation

2. **No Labels**: We don't have ground truth for pause predictions
   - Can't directly supervise "this should be a pause" vs "this shouldn't"
   - Would need to create synthetic labels or use indirect methods

3. **Complex Behavior**: Pause prediction is subtle
   - Legitimate pauses vs. digit-induced false positives
   - Risk of breaking normal pause detection

#### Theoretical Implementation (If Pursued)

**Would require**:
1. Modify `train.py` to compute extra heads during forward pass
2. Create synthetic pause labels for training data
3. Add extra heads loss to total loss
4. Extensive testing to avoid breaking normal VAD

**Dataset Requirements**:
- Audio samples with **ground truth pause annotations**
- Explicitly labeled: "no pause should occur here" during digit sequences
- Much larger dataset needed (1000+ samples minimum)

**Code modifications needed**:
```python
# In train.py, after line 255:
output = model(codes=codes, condition_tensors=condition_tensors)

# Add:
if hasattr(model, 'extra_heads') and len(model.extra_heads) > 0:
    # Compute extra heads output
    # Would need transformer_out from model
    # Add loss computation comparing to ground truth pause labels
    # This is complex and not currently supported
```

#### Pros & Cons

✅ **Pros**:
- Preserves digit transcription format
- Directly addresses the root cause (pause prediction)

❌ **Cons**:
- Much more complex implementation
- Requires modifying training loop significantly
- No ground truth pause labels
- Risk of breaking normal VAD functionality
- Requires deep understanding of pause prediction mechanism
- Larger dataset needed
- Harder to validate success

---

### Approach 3: Post-Processing Conversion (No Training)

**Strategy**: Keep model as-is, convert word-form numerals back to digits in post-processing.

#### Implementation

If you choose Approach 1 but need digit output:

```python
import re

WORD_TO_DIGIT = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}

def convert_number_words_to_digits(text, context_aware=True):
    """
    Convert word-form numerals to digits.

    Args:
        text: Input text with word-form numbers
        context_aware: If True, only convert in likely number sequence contexts
    """
    if context_aware:
        # Only convert in contexts like phone numbers, zip codes, etc.
        # More sophisticated: check if multiple number words in sequence
        pattern = r'\b(zero|one|two|three|four|five|six|seven|eight|nine)\b'

        def convert_sequence(match_text):
            # Check if we have multiple number words in close proximity
            words = match_text.split()
            number_words = [w for w in words if w.lower() in WORD_TO_DIGIT]

            # If 3+ consecutive number words, likely a sequence
            if len(number_words) >= 3:
                return ' '.join(WORD_TO_DIGIT.get(w.lower(), w) for w in words)
            return match_text

        # More complex implementation needed for context awareness
        return text
    else:
        # Simple: convert all number words
        for word, digit in WORD_TO_DIGIT.items():
            text = re.sub(r'\b' + word + r'\b', digit, text, flags=re.IGNORECASE)
        return text

# Example usage
transcription = "My phone number is three one zero five five five one two three four"
converted = convert_number_words_to_digits(transcription, context_aware=False)
print(converted)  # "My phone number is 3 1 0 5 5 5 1 2 3 4"
```

#### Pros & Cons

✅ **Pros**:
- No training required
- Can be refined with rules
- Flexible (can toggle on/off)

❌ **Cons**:
- Still requires Approach 1 to fix VAD issue
- Extra processing step
- Ambiguity in context (when to convert vs. not)

---

## Recommended Path Forward

### Phase 1: Implement Approach 1 (Word-Form Transcription)

1. **Collect minimal dataset** (50-100 samples):
   - Focus on common scenarios: phone numbers, zip codes, addresses
   - 3-10 seconds per sample
   - Mix of speakers if possible

2. **Prepare transcriptions**:
   - Use Whisper for initial transcription
   - Manually review and convert digits to words
   - Ensure consistency (always "three one zero", not "three hundred ten")

3. **Train initial model**:
   - Use provided config
   - 300-500 steps
   - Monitor loss (target: < 0.3)

4. **Test VAD behavior**:
   - Record test cases with number sequences
   - Verify transcription uses word-form
   - **Critical**: Verify pause prediction doesn't spike during numbers

5. **Iterate**:
   - If still issues, add more targeted examples
   - Increase training steps
   - Adjust learning rate if needed

### Phase 2: Convert to Candle Format

Once fine-tuning is successful:

1. **Export LoRA weights**:
   ```bash
   # Weights are in: ./output/stt_numbers_finetune/checkpoints/checkpoint_XXXXX/consolidated/lora.safetensors
   ```

2. **Merge LoRA into base model** (if needed):
   ```python
   # In Python, load base model + LoRA and save merged weights
   # This may be required for Candle conversion
   ```

3. **Convert to Candle**:
   - Research Candle model format requirements
   - May need to use Candle's model conversion tools
   - Ensure STT config (dep_q=0) is preserved

4. **Test in your application**:
   - Verify transcription quality
   - Verify VAD behavior in production environment

### Phase 3 (Optional): Post-Processing

If your application requires digit format output:
- Implement post-processing conversion (Approach 3)
- Add context-aware rules based on your use case
- Test edge cases

## Dataset Collection Tips

### Recording Scenarios

**Phone Numbers**:
- "My phone number is [sequence]"
- "Call me at [sequence]"
- "You can reach me at [sequence]"

**Zip Codes**:
- "My zip code is [sequence]"
- "I live in [sequence]"

**Addresses**:
- "The address is [sequence] [street name]"
- "I'm at [sequence] Main Street"

**Counting/Lists**:
- "One, two, three, four, five..."
- "The items are numbered one through ten"

**Mixed Context**:
- "I need two pizzas delivered to three one zero five Main Street"
- Helps ensure model doesn't overcorrect non-sequence numbers

### Quality Guidelines

- **Audio Quality**: Clear speech, minimal background noise
- **Pacing**: Natural speaking pace (not too slow or fast)
- **Variety**: Different speakers, accents, speaking styles
- **Length**: 3-10 seconds optimal (fits within `duration_sec: 30`)
- **Format**: Mono, 16kHz sample rate, WAV format

### Transcription Guidelines

- **Consistency**: Always use word-form for digits (0-9)
- **Spacing**: "three one zero" (with spaces), not "threeonezerо"
- **Punctuation**: Minimal; add periods at sentence end
- **No normalization**: Don't convert "three one zero" to "310" at this stage

## Testing & Validation

### Success Criteria

1. **Transcription Accuracy**: Model outputs word-form numerals
2. **VAD Behavior**: Pause prediction doesn't spike during number sequences
3. **General Performance**: Model still handles non-number speech correctly
4. **Consistency**: Reliable behavior across different speakers/contexts

### Test Cases

```python
test_cases = [
    {
        "audio": "phone_number_1.wav",
        "expected": "my phone number is three one zero five five five one two three four",
        "should_not_cut_off": True
    },
    {
        "audio": "zip_code_1.wav",
        "expected": "the zip code is nine zero two one zero",
        "should_not_cut_off": True
    },
    {
        "audio": "normal_speech_1.wav",
        "expected": "hello how are you doing today",
        "vad_should_work_normally": True
    }
]
```

### Monitoring During Training

Watch for:
- **Loss convergence**: Should drop steadily to < 0.5
- **Overfitting**: If loss drops to ~0 on small dataset, you may be overfitting
  - Symptom: Perfect on training samples, fails on new number sequences
  - Solution: Add more diverse examples, reduce training steps
- **Learning rate issues**: If loss oscillates, reduce LR
- **Memory issues**: Reduce batch_size or duration_sec if OOM

## Cost-Benefit Analysis

| Aspect | Approach 1 (Words) | Approach 2 (Extra Heads) | Approach 3 (Post-Process) |
|--------|-------------------|------------------------|--------------------------|
| **Implementation Complexity** | Low | Very High | Minimal |
| **Training Time** | Medium | Long | None |
| **Dataset Size** | 50-500 samples | 1000+ samples | N/A |
| **Risk of Breaking VAD** | Low | High | None |
| **Output Format** | Words | Digits | Digits |
| **Success Probability** | High | Medium | N/A |
| **Development Time** | 1-2 weeks | 4-8 weeks | 1 day |

## Conclusion

**Recommended Approach**: Implement Approach 1 (word-form transcription training) with optional Approach 3 (post-processing conversion) if digit output is required.

This approach:
- Has highest probability of success
- Lowest implementation complexity
- Manageable dataset requirements
- Low risk of breaking existing functionality
- Can be completed in reasonable timeframe

The key insight is that the model already handles word-form numerals correctly for VAD—we just need to teach it to consistently use that format.
