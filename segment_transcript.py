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

@dataclass
class ChapterMarker:
    timestamp: str
    title: str
    description: str
    significance_score: float  # 0.0-1.0, how major this transition is

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

def parse_transcript(file_path: Path) -> List[dict]:
    """Parse either raw or enhanced transcript format into utterances"""
    content = file_path.read_text()
    lines = content.strip().split('\n')
    
    # Detect format by looking at first few lines
    format_type = detect_transcript_format(lines)
    
    if format_type == 'enhanced':
        return parse_enhanced_transcript(lines)
    else:
        return parse_raw_transcript_lines(lines)

def detect_transcript_format(lines: List[str]) -> str:
    """Detect if this is raw or enhanced transcript format"""
    for line in lines[:20]:  # Check first 20 lines
        line = line.strip()
        if not line:
            continue
        
        # Enhanced format: "Speaker (HH:MM:SS): text on same line"
        if '):' in line and '(' in line:
            # Look for pattern: Name (timestamp): text
            import re
            if re.match(r'^[^()]+\([0-9:]+\):\s*.+', line):
                return 'enhanced'
        
        # Raw format: "Speaker HH:MM:SS" on separate line from text
        parts = line.split()
        if len(parts) >= 2:
            timestamp_candidate = parts[-1]
            if ':' in timestamp_candidate and len(timestamp_candidate.split(':')) == 3:
                try:
                    time_parts = timestamp_candidate.split(':')
                    int(time_parts[0])
                    int(time_parts[1])
                    int(time_parts[2])
                    return 'raw'
                except ValueError:
                    continue
    
    return 'raw'  # Default to raw format

def parse_enhanced_transcript(lines: List[str]) -> List[dict]:
    """Parse enhanced transcript format: Name (HH:MM:SS): text"""
    import re
    utterances = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match pattern: "Speaker Name (HH:MM:SS): text"
        match = re.match(r'^([^()]+)\(([0-9:]+)\):\s*(.+)', line)
        if match:
            speaker = match.group(1).strip()
            timestamp = match.group(2)
            text = match.group(3).strip()
            
            if speaker and timestamp and text:
                utterances.append({
                    'speaker': speaker,
                    'timestamp': timestamp,
                    'text': text,
                    'seconds': parse_timestamp_to_seconds(timestamp)
                })
    
    return utterances

def parse_raw_transcript_lines(lines: List[str]) -> List[dict]:
    """Parse raw transcript format: Speaker HH:MM:SS on separate lines"""
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
                    current_speaker = ' '.join(parts[:-1])  # Everything except the timestamp
                    current_time = timestamp_candidate       # The timestamp
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
    model = generativeai.GenerativeModel('gemini-2.5-flash')
    
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

def identify_chapter_markers(segments: List[TranscriptSegment], api_key: str, target_chapters: int = 12) -> List[ChapterMarker]:
    """Analyze segments to identify 10-15 major chapter markers for bookmarks"""
    if not segments:
        return []
    
    generativeai.configure(api_key=api_key)
    model = generativeai.GenerativeModel('gemini-2.5-flash')
    
    # Create overview of all segments for context
    segments_overview = "\n".join([
        f"{i+1}. {seg.start_time} - {seg.topic}: {seg.summary}"
        for i, seg in enumerate(segments)
    ])
    
    # Get total duration for context
    total_duration = parse_timestamp_to_seconds(segments[-1].end_time) if segments else 0
    total_minutes = total_duration // 60
    
    prompt = f"""You are creating chapter markers for a podcast episode to help listeners navigate to major sections.

Episode overview ({total_minutes} minutes):
{segments_overview}

Identify {target_chapters} major chapter markers that represent the most significant topic transitions and story beats. These will be used as bookmarks/chapters for listeners.

Criteria for good chapter markers:
1. Major topic shifts or story transitions
2. Natural narrative breaks
3. Introduction of new concepts or people
4. Key turning points in the story
5. Evenly distributed throughout the episode (avoid clustering)
6. Would be useful navigation points for listeners

For each chapter marker, provide:
- The timestamp where this new chapter begins
- A compelling chapter title (3-8 words, like a book chapter)
- Brief description of what this section covers
- Significance score (0.0-1.0) - how major/important this transition is

Return as JSON array:
[{{
  "timestamp": "HH:MM:SS",
  "title": "Compelling Chapter Title",
  "description": "What this section covers",
  "significance_score": 0.8
}}]

Focus on the most significant {target_chapters} transitions that would help listeners navigate the episode."""
    
    try:
        response = model.generate_content(prompt)
        
        # Parse JSON response
        import json
        import re
        json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
        
        if json_match:
            markers_data = json.loads(json_match.group())
            chapter_markers = []
            
            for marker in markers_data:
                chapter_markers.append(ChapterMarker(
                    timestamp=marker['timestamp'],
                    title=marker['title'],
                    description=marker['description'],
                    significance_score=float(marker.get('significance_score', 0.5))
                ))
            
            # Sort by timestamp and return
            return sorted(chapter_markers, key=lambda x: parse_timestamp_to_seconds(x.timestamp))
    
    except Exception as e:
        print(f"Error identifying chapter markers: {e}")
        
        # Fallback: Create markers from highest-scoring segments
        print("Using fallback method to create chapter markers...")
        
        # Select segments based on position and significance
        markers = []
        segments_per_chapter = max(1, len(segments) // target_chapters)
        
        for i in range(0, len(segments), segments_per_chapter):
            if len(markers) >= target_chapters:
                break
            seg = segments[i]
            markers.append(ChapterMarker(
                timestamp=seg.start_time,
                title=seg.topic,
                description=seg.summary,
                significance_score=0.6
            ))
        
        return markers

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
    utterances = parse_transcript(transcript_path)
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
    
    # Identify chapter markers
    print(f"\nIdentifying chapter markers...")
    chapter_markers = identify_chapter_markers(segments, api_key)
    
    print(f"\n=== Found {len(chapter_markers)} Chapter Markers ===\n")
    for i, marker in enumerate(chapter_markers, 1):
        print(f"Chapter {i}: {marker.title}")
        print(f"  Timestamp: {marker.timestamp}")
        print(f"  Description: {marker.description}")
        print(f"  Significance: {marker.significance_score:.1f}/1.0")
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
        
        # Export chapter markers to separate CSV
        if chapter_markers:
            chapters_path = output_path.with_name(output_path.stem + '_chapters' + output_path.suffix)
            with open(chapters_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Chapter", "Timestamp", "Title", "Description", "Significance Score"])
                
                for i, marker in enumerate(chapter_markers, 1):
                    writer.writerow([
                        i,
                        marker.timestamp,
                        marker.title,
                        marker.description,
                        marker.significance_score
                    ])
            
            print(f"📖 Chapter markers exported to: {chapters_path}")

if __name__ == "__main__":
    main()