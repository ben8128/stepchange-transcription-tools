#!/usr/bin/env python3
"""
Enhance raw transcripts for readability and clarity.
Creates a clean, shareable version while preserving the original.
"""
import argparse
from pathlib import Path
import os
from google import generativeai
from dataclasses import dataclass
from typing import List

@dataclass
class Utterance:
    """A single utterance from a speaker"""
    speaker: str
    timestamp: str
    text: str
    
def parse_raw_transcript(file_path: Path) -> List[Utterance]:
    """Parse a raw transcript file into Utterance objects"""
    content = file_path.read_text()
    lines = content.strip().split('\n')
    
    utterances = []
    current_speaker = None
    current_time = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a speaker/timestamp line (format: "SpeakerName HH:MM:SS")
        parts = line.split()
        if len(parts) >= 2:
            # Check if last part looks like a timestamp (HH:MM:SS format)
            timestamp_candidate = parts[-1]
            if ':' in timestamp_candidate and len(timestamp_candidate.split(':')) == 3:
                try:
                    # Validate it's actually a timestamp by parsing
                    time_parts = timestamp_candidate.split(':')
                    int(time_parts[0])  # hours
                    int(time_parts[1])  # minutes  
                    int(time_parts[2])  # seconds
                    # If we get here, it's a valid timestamp format
                    
                    # Save previous utterance if exists
                    if current_speaker and current_text:
                        text = ' '.join(current_text).strip()
                        if text:
                            utterances.append(Utterance(
                                speaker=current_speaker,
                                timestamp=current_time,
                                text=text
                            ))
                    
                    # Start new utterance
                    current_speaker = ' '.join(parts[:-1])  # Everything except the timestamp
                    current_time = timestamp_candidate      # The timestamp
                    current_text = []
                    continue
                    
                except ValueError:
                    # Not a valid timestamp, treat as regular text
                    pass
        
        # Add text to current utterance
        if current_speaker:
            current_text.append(line)
    
    # Don't forget the last utterance
    if current_speaker and current_text:
        text = ' '.join(current_text).strip()
        if text:
            utterances.append(Utterance(
                speaker=current_speaker,
                timestamp=current_time,
                text=text
            ))
    
    return utterances

class Enhancer:
    """Handles enhancing transcripts using Gemini"""
    
    PROMPT = """You are an expert transcript editor. Your task is to enhance this transcript for maximum readability.

Rules:
1. Fix ONLY clear errors (grammar, filler words, false starts)
2. Preserve the natural conversation flow 
3. Keep the speaker's unique voice and style
4. Never add or invent content
5. Keep technical terms and proper nouns accurate
6. PRESERVE all timestamps exactly as provided
7. Format speaker labels as "SpeakerName (HH:MM:SS):" using ONLY the speaker name (no "Speaker" prefix, no bold formatting)
8. Add natural paragraph breaks for readability
9. Fix run-on sentences by adding appropriate punctuation

Common fixes to make:
- Remove filler words like "um", "uh", "you know" when they don't add meaning
- Fix clear grammatical errors
- Add punctuation where obviously missing
- Fix word repetitions from false starts
- Correct obvious transcription errors

Output format:
SpeakerName (00:01:23): [enhanced text with proper capitalization and punctuation]

CRITICAL: Use ONLY the speaker name without any prefix or formatting:
- CORRECT: "Ben Shwab Eidelson (00:01:23): Enhanced text here."
- WRONG: "**Speaker Ben Shwab Eidelson (00:01:23):** Enhanced text here."
- WRONG: "Speaker Ben Shwab Eidelson (00:01:23): Enhanced text here."

Use the exact speaker names from the input transcript, but remove any "Speaker" prefix if present.

IMPORTANT: Keep every timestamp exactly as provided. Do not modify, remove, or approximate timestamps.

Remember: The goal is readability while maintaining authenticity and precise timing information."""

    def __init__(self, api_key: str):
        generativeai.configure(api_key=api_key)
        self.model = generativeai.GenerativeModel("gemini-2.5-flash")
    
    def enhance_transcript(self, utterances: List[Utterance]) -> str:
        """Enhance the transcript for readability"""
        # Process in chunks to avoid token limits
        chunk_size = 50  # utterances per chunk
        enhanced_chunks = []
        
        for i in range(0, len(utterances), chunk_size):
            chunk = utterances[i:i+chunk_size]
            
            # Format chunk for enhancement
            raw_text = "\n\n".join([
                f"{u.speaker} ({u.timestamp}): {u.text}"
                for u in chunk
            ])
            
            print(f"Enhancing utterances {i+1}-{min(i+chunk_size, len(utterances))}...")
            
            try:
                response = self.model.generate_content([self.PROMPT, raw_text])
                enhanced_chunks.append(response.text)
            except Exception as e:
                print(f"Error enhancing chunk: {e}")
                # Fallback to original if enhancement fails
                fallback = "\n\n".join([
                    f"{u.speaker} ({u.timestamp}): {u.text}"
                    for u in chunk
                ])
                enhanced_chunks.append(fallback)
        
        # Combine all enhanced chunks
        enhanced_text = "\n\n---\n\n".join(enhanced_chunks)
        
        # Post-process to ensure consistent formatting
        enhanced_text = self.clean_speaker_formatting(enhanced_text)
        
        return enhanced_text
    
    def clean_speaker_formatting(self, text: str) -> str:
        """Clean up any inconsistent speaker formatting"""
        import re
        
        # Remove "Speaker " prefix and bold formatting
        # Patterns: **Speaker Name (timestamp):** or Speaker Name (timestamp):
        text = re.sub(r'\*\*Speaker ([^(]+) \(([^)]+)\):\*\*', r'\1 (\2):', text)
        text = re.sub(r'Speaker ([^(]+) \(([^)]+)\):', r'\1 (\2):', text)
        
        # Clean up any remaining ** formatting around speaker labels
        text = re.sub(r'\*\*([^*]+) \(([^)]+)\):\*\*', r'\1 (\2):', text)
        
        return text

def main():
    parser = argparse.ArgumentParser(description="Enhance raw transcripts for readability")
    parser.add_argument("raw_transcript", help="Path to the raw transcript file")
    parser.add_argument("--output", help="Output path for enhanced transcript (default: adds .enhanced.md)")
    parser.add_argument("--google-api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    raw_path = Path(args.raw_transcript)
    if not raw_path.exists():
        print(f"File not found: {raw_path}")
        return
    
    # Get API key
    api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY found. Set it via --google-api-key or environment variable.")
        return
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Replace .raw.md with .enhanced.md, or just add .enhanced.md
        if raw_path.name.endswith('.raw.md'):
            output_path = raw_path.with_suffix('').with_suffix('.enhanced.md')
        else:
            output_path = raw_path.with_suffix('.enhanced.md')
    
    print(f"Loading raw transcript: {raw_path}")
    utterances = parse_raw_transcript(raw_path)
    print(f"Parsed {len(utterances)} utterances")
    
    print("\nEnhancing transcript for readability...")
    enhancer = Enhancer(api_key)
    enhanced = enhancer.enhance_transcript(utterances)
    
    # Save enhanced transcript
    output_path.write_text(enhanced)
    print(f"\n✨ Enhanced transcript saved to: {output_path}")
    
    # Show sample comparison
    print("\n=== Sample Enhancement ===")
    print("Original (first utterance):")
    if utterances:
        print(f"  {utterances[0].speaker}: {utterances[0].text[:150]}...")
    print("\nEnhanced version saved to file for review.")

if __name__ == "__main__":
    main()