# Recording Script for STT Number Training Dataset

## Instructions

- Record each sample as a separate mono WAV file
- Speak naturally at a normal conversational pace
- Keep each recording 3-10 seconds long
- Name files: `sample_001.wav`, `sample_002.wav`, etc.
- Try to vary your tone and delivery across samples
- If possible, have multiple people record some samples for variety

## Sample Scripts

### Order Numbers (7 digits) - Samples 1-15

**Sample 1:**
"The order number is one two nine six eight zero three."

**Sample 2:**
"I'm calling about order number three four five six seven eight nine."

**Sample 3:**
"Can you look up order one one two two three three four?"

**Sample 4:**
"My order number is seven eight nine zero one two three."

**Sample 5:**
"I need to track order number five five five one two three four."

**Sample 6:**
"The order I placed is number two zero zero nine eight seven six."

**Sample 7:**
"Order number nine nine nine eight eight eight seven."

**Sample 8:**
"I'm checking on order four four four five five five six."

**Sample 9:**
"Can you find order number six seven eight nine zero one two?"

**Sample 10:**
"My order is eight eight eight seven seven seven six."

**Sample 11:**
"I have a question about order one three five seven nine two four."

**Sample 12:**
"The order number is three three three two two two one."

**Sample 13:**
"Please check order number nine eight seven six five four three."

**Sample 14:**
"I'm looking for order eight zero eight zero eight zero eight."

**Sample 15:**
"Order number two four six eight zero two four, please."

### Location Codes (6 digits) - Samples 16-30

**Sample 16:**
"The location code is one four five six one nine."

**Sample 17:**
"I'm at location code three three three four four four."

**Sample 18:**
"Can you check location two zero two zero two zero?"

**Sample 19:**
"My location code is nine eight seven six five four."

**Sample 20:**
"The store location is one one one two two two."

**Sample 21:**
"I'm calling from location code five five five six six six."

**Sample 22:**
"The delivery location is seven eight nine one two three."

**Sample 23:**
"Location code four four four five five five, please."

**Sample 24:**
"I'm at location eight zero zero nine zero zero."

**Sample 25:**
"The pickup location is code six six six seven seven seven."

**Sample 26:**
"My location is one two three four five six."

**Sample 27:**
"Location code nine nine eight eight seven seven."

**Sample 28:**
"I need to go to location two four six eight one three."

**Sample 29:**
"The warehouse location is three six nine two five eight."

**Sample 30:**
"I'm at location code seven seven seven eight eight eight."

### Phone Numbers - Samples 31-40

**Sample 31:**
"My phone number is three one zero five five five one two three four."

**Sample 32:**
"You can reach me at four one five two two two three three three three."

**Sample 33:**
"Call me at eight one eight seven seven seven eight eight eight eight."

**Sample 34:**
"My number is six five zero three three three four four four four."

**Sample 35:**
"I can be reached at nine one seven five five five six six six six."

**Sample 36:**
"The contact number is two one two eight eight eight nine nine nine nine."

**Sample 37:**
"My cell phone is seven zero two four four four five five five five."

**Sample 38:**
"Please call five five five one two three four five six seven."

**Sample 39:**
"My phone is six one nine three three three two two two two."

**Sample 40:**
"You can call me at three zero three seven seven seven eight eight eight eight."

### Mixed Context - Samples 41-50

**Sample 41:**
"I have order number one two three four five six seven for location two two two three three three."

**Sample 42:**
"Please send it to location code eight eight eight nine nine nine, my phone is three one zero five five five one two three four."

**Sample 43:**
"The order is seven seven seven eight eight eight nine and my callback number is four one five two two two three three three three."

**Sample 44:**
"I need to update order three four five six seven eight nine with location one one one two two two."

**Sample 45:**
"My order number is nine nine nine zero zero zero one and the location is five five five six six six."

**Sample 46:**
"Can you ship order two two two three three three four to location code seven eight nine zero one two?"

**Sample 47:**
"I'm at location one four five six one nine calling about order eight eight eight seven seven seven six."

**Sample 48:**
"Order number five five five four four four three needs to go to location nine eight seven six five four."

**Sample 49:**
"Please call me at six five zero three three three four four four four about order one two nine six eight zero three."

**Sample 50:**
"My order is four four four five five five six for pickup at location code three three three two two two one."

---

## Expected Transcription Format

When you create the JSON transcription files, use this format with **word-form numerals**:

### Example for Sample 1:
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.5,
      "text": "The order number is one two nine six eight zero three."
    }
  ]
}
```

### Example for Sample 41 (multiple segments):
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 7.2,
      "text": "I have order number one two three four five six seven for location two two two three three three."
    }
  ]
}
```

## Recording Tips

1. **Environment**: Record in a quiet space with minimal background noise
2. **Equipment**: Use a decent microphone if available (phone mic is okay for testing)
3. **Distance**: Speak 6-12 inches from the microphone
4. **Volume**: Speak at normal conversational volume (not too loud or soft)
5. **Pace**: Natural speaking speed - don't rush or speak too slowly
6. **Clarity**: Pronounce each digit clearly, especially zero vs. "oh"
7. **Consistency**: Always say "zero" not "oh" for 0
8. **Variation**: Try different tones - some formal, some casual, some faster, some slower

## Audio Processing

After recording, ensure all files are:
- **Format**: WAV (16-bit PCM)
- **Channels**: Mono
- **Sample Rate**: 16000 Hz (16 kHz) recommended

Use this command to convert if needed:
```bash
sox input.wav -r 16000 -c 1 -b 16 output.wav
```

## Next Steps After Recording

1. Save all recordings as `sample_001.wav` through `sample_050.wav`
2. Create the dataset JSON index file (`train.jsonl`)
3. Generate transcription files using `annotate.py` OR manually create them
4. Manually review and correct transcriptions to ensure word-form numerals
5. Validate dataset structure
6. Begin training!
