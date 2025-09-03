#!/usr/bin/env python3
"""
Segment transcript into story segments based on topic changes and natural breaks.
Outputs timestamps and summaries for each segment.
"""
import argparse
from pathlib import Path
import csv
from dataclasses import dataclass
from typing import List
import os
from google import generativeai

@dataclass
class TranscriptSegment:
    start_time: str
    end_time: str
    duration_seconds: int
    topic: str
    summary: str
    speakers: str
    key_points: List[str]

def parse_timestamp_to_seconds(timestamp_str: str) -> int:
    """Convert HH:MM:SS timestamp to seconds"""
    try:
        parts = timestamp_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        else:
            return 0
    except (ValueError, AttributeError):
        return 0

def seconds_to_timestamp(seconds: int) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def parse_raw_transcript(file_path: Path) -> List[dict]:
    """Parse a raw transcript file into utterances"""
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
            
        # Check if this is a speaker/timestamp line
        if len(line.split()) == 2:
            parts = line.split()
            if len(parts[0]) == 1 and ':' in parts[1]:
                # Save previous utterance if exists
                if current_speaker and current_text:
                    text = ' '.join(current_text).strip()
                    if text:
                        utterances.append({
                            'speaker': current_speaker,
                            'timestamp': current_time,
                            'text': text,
                            'seconds': parse_timestamp_to_seconds(current_time)
                        })
                
                # Start new utterance
                current_speaker = parts[0]
                current_time = parts[1]
                current_text = []
                continue
        
        # Add text to current utterance
        if current_speaker:
            current_text.append(line)
    
    # Don't forget the last utterance
    if current_speaker and current_text:
        text = ' '.join(current_text).strip()
        if text:
            utterances.append({
                'speaker': current_speaker,
                'timestamp': current_time,
                'text': text,
                'seconds': parse_timestamp_to_seconds(current_time)
            })
    
    return utterances

def segment_with_ai(utterances: List[dict], api_key: str, window_size: int = 10) -> List[TranscriptSegment]:
    """Use AI to identify natural story segments and topic changes"""
    generativeai.configure(api_key=api_key)
    model = generativeai.GenerativeModel('gemini-2.0-flash-exp')
    
    segments = []
    
    # Process in windows to identify segment boundaries
    for i in range(0, len(utterances), window_size):
        window = utterances[i:min(i+window_size*2, len(utterances))]
        
        # Create context for segmentation
        context = "\n".join([
            f"{u['timestamp']} {u['speaker']}: {u['text'][:200]}..."
            for u in window
        ])
        
        prompt = f"""Analyze this transcript section and identify distinct story segments or topic changes.
For each segment, provide:
1. The starting timestamp
2. The ending timestamp
3. A brief topic title (5-10 words)
4. A one-sentence summary
5. Key points (2-3 bullet points)

Transcript:
{context}

Return as JSON array with format:
[{{
  "start": "HH:MM:SS",
  "end": "HH:MM:SS", 
  "topic": "Brief Topic Title",
  "summary": "One sentence summary",
  "key_points": ["point 1", "point 2"]
}}]"""

        try:
            response = model.generate_content(prompt)
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            text = response.text
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                segment_data = json.loads(json_match.group())
                
                for seg in segment_data:
                    # Calculate duration
                    start_sec = parse_timestamp_to_seconds(seg['start'])
                    end_sec = parse_timestamp_to_seconds(seg['end'])
                    
                    # Determine speakers in this segment
                    speakers_in_segment = set()
                    for u in window:
                        if start_sec <= u['seconds'] <= end_sec:
                            speakers_in_segment.add(u['speaker'])
                    
                    segments.append(TranscriptSegment(
                        start_time=seg['start'],
                        end_time=seg['end'],
                        duration_seconds=end_sec - start_sec,
                        topic=seg['topic'],
                        summary=seg['summary'],
                        speakers=', '.join(sorted(speakers_in_segment)) or 'A, B',
                        key_points=seg.get('key_points', [])
                    ))
        except Exception as e:
            print(f"Error processing window {i}: {e}")
            continue
    
    # Merge overlapping segments and remove duplicates
    merged = []
    for seg in sorted(segments, key=lambda x: parse_timestamp_to_seconds(x.start_time)):
        if not merged or parse_timestamp_to_seconds(seg.start_time) > parse_timestamp_to_seconds(merged[-1].end_time):
            merged.append(seg)
    
    return merged

def main():
    parser = argparse.ArgumentParser(description="Segment transcript into story sections")
    parser.add_argument("transcript_file", help="Path to the raw transcript file")
    parser.add_argument("--window-size", type=int, default=10, help="Utterances per analysis window (default: 10)")
    parser.add_argument("--export-csv", help="Export segments to CSV file")
    parser.add_argument("--google-api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    transcript_path = Path(args.transcript_file)
    if not transcript_path.exists():
        print(f"File not found: {transcript_path}")
        return
    
    # Get API key
    api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY found. Set it via --google-api-key or environment variable.")
        return
    
    print(f"Loading transcript: {transcript_path}")
    utterances = parse_raw_transcript(transcript_path)
    print(f"Parsed {len(utterances)} utterances")
    
    print(f"\nSegmenting transcript into story sections...")
    segments = segment_with_ai(utterances, api_key, args.window_size)
    
    print(f"\n=== Found {len(segments)} Story Segments ===\n")
    
    for i, seg in enumerate(segments, 1):
        print(f"Segment {i}: {seg.topic}")
        print(f"  Time: {seg.start_time} - {seg.end_time} ({seg.duration_seconds}s)")
        print(f"  Speakers: {seg.speakers}")
        print(f"  Summary: {seg.summary}")
        if seg.key_points:
            print(f"  Key Points:")
            for point in seg.key_points:
                print(f"    • {point}")
        print()
    
    # Export to CSV if requested
    if args.export_csv:
        output_path = Path(args.export_csv)
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Segment", "Start Time", "End Time", "Duration (s)", "Topic", "Summary", "Speakers", "Key Points"])
            
            for i, seg in enumerate(segments, 1):
                key_points_str = " | ".join(seg.key_points) if seg.key_points else ""
                writer.writerow([
                    i,
                    seg.start_time,
                    seg.end_time,
                    seg.duration_seconds,
                    seg.topic,
                    seg.summary,
                    seg.speakers,
                    key_points_str
                ])
        
        print(f"📊 Segments exported to: {output_path}")

if __name__ == "__main__":
    main()