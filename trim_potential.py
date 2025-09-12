#!/usr/bin/env python3
"""
Analyze transcript to identify potential content for removal to achieve target duration.
Identifies redundant sections, non-essential tangents, and less critical content.
"""
import argparse
from pathlib import Path
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
from google import generativeai
import re

@dataclass
class TrimCandidate:
    segment_id: str
    start_time: str
    end_time: str
    duration_seconds: int
    trim_type: str  # 'redundant', 'tangent', 'non-essential', 'repetitive'
    confidence: float  # 0.0-1.0 how safe it is to remove
    reason: str
    text_preview: str
    impact_on_flow: str  # 'minimal', 'moderate', 'significant'

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

def parse_raw_transcript(file_path: Path) -> List[Dict]:
    """Parse raw transcript into segments with timestamps and speakers"""
    content = file_path.read_text()
    lines = content.strip().split('\n')
    
    segments = []
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
                    # Validate timestamp
                    time_parts = timestamp_candidate.split(':')
                    int(time_parts[0])
                    int(time_parts[1])
                    int(time_parts[2])
                    
                    # Save previous segment
                    if current_speaker and current_text:
                        text = ' '.join(current_text).strip()
                        if text:
                            segments.append({
                                'speaker': current_speaker,
                                'timestamp': current_time,
                                'text': text,
                                'seconds': parse_timestamp_to_seconds(current_time)
                            })
                    
                    # Start new segment
                    current_speaker = ' '.join(parts[:-1])  # Everything except the timestamp
                    current_time = timestamp_candidate       # The timestamp
                    current_text = []
                    continue
                except ValueError:
                    pass
        
        # Add text to current segment
        if current_speaker:
            current_text.append(line)
    
    # Don't forget the last segment
    if current_speaker and current_text:
        text = ' '.join(current_text).strip()
        if text:
            segments.append({
                'speaker': current_speaker,
                'timestamp': current_time,
                'text': text,
                'seconds': parse_timestamp_to_seconds(current_time)
            })
    
    return segments

def load_story_segments(segments_csv: Path) -> List[Dict]:
    """Load story segments from CSV"""
    segments = []
    if segments_csv.exists():
        with open(segments_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                segments.append({
                    'segment_num': row['Segment'],
                    'start_time': row['Start Time'],
                    'end_time': row['End Time'],
                    'duration': int(row['Duration (s)']),
                    'topic': row['Topic'],
                    'summary': row['Summary'],
                    'speakers': row['Speakers'],
                    'key_points': row['Key Points'].split(' | ') if row['Key Points'] else []
                })
    return segments

class TrimAnalyzer:
    """AI-powered analyzer to identify trim candidates"""
    
    def __init__(self, api_key: str):
        generativeai.configure(api_key=api_key)
        self.model = generativeai.GenerativeModel("gemini-2.5-flash")
    
    def analyze_segments_for_trimming(self, transcript_segments: List[Dict], 
                                    story_segments: List[Dict], 
                                    target_minutes: int = 30) -> List[TrimCandidate]:
        """Analyze segments to identify trim candidates"""
        
        trim_candidates = []
        
        # Process story segments in batches
        batch_size = 10
        for i in range(0, len(story_segments), batch_size):
            batch = story_segments[i:i+batch_size]
            
            # Create analysis prompt
            segments_text = "\n\n".join([
                (f"Segment {seg['segment_num']}: {seg['topic']} ({seg['duration']}s)\n" +
                 f"Time: {seg['start_time']} - {seg['end_time']}\n" +
                 f"Summary: {seg['summary']}\n" +
                 f"Key Points: {'; '.join(seg['key_points'])}")
                for seg in batch
            ])
            
            prompt = f"""Analyze these podcast segments to identify content that could be trimmed to reduce episode length by ~{target_minutes} minutes.

Segments to analyze:
{segments_text}

Identify segments that are:
1. REDUNDANT: Similar points made elsewhere 
2. TANGENTIAL: Interesting but not core to the main story
3. NON-ESSENTIAL: Could be removed without harming narrative flow
4. REPETITIVE: Rehashing points already covered

For each trim candidate, provide:
- Segment number
- Trim type (redundant/tangent/non-essential/repetitive)
- Confidence (0.0-1.0) that it's safe to remove
- Reason for removal
- Impact on flow (minimal/moderate/significant)

Return as JSON array:
[{{
  "segment_num": "1",
  "trim_type": "tangent",
  "confidence": 0.8,
  "reason": "Watson's alcohol incident is colorful but tangential to data center story",
  "impact_on_flow": "minimal"
}}]

Focus on segments that:
- Don't advance the core narrative
- Repeat information covered elsewhere
- Are interesting asides but not essential
- Could be condensed significantly

Prioritize higher confidence (safer) removals first."""
            
            try:
                response = self.model.generate_content(prompt)
                
                # Parse JSON response
                import json
                json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
                if json_match:
                    candidates_data = json.loads(json_match.group())
                    
                    for candidate in candidates_data:
                        # Find the corresponding story segment
                        seg_num = candidate['segment_num']
                        story_seg = next((s for s in batch if s['segment_num'] == seg_num), None)
                        
                        if story_seg:
                            # Get text preview from transcript
                            start_sec = parse_timestamp_to_seconds(story_seg['start_time'])
                            preview_segments = [s for s in transcript_segments 
                                             if abs(s['seconds'] - start_sec) < 30]
                            text_preview = ' '.join([s['text'][:100] for s in preview_segments[:2]])
                            
                            trim_candidates.append(TrimCandidate(
                                segment_id=f"Segment_{seg_num}",
                                start_time=story_seg['start_time'],
                                end_time=story_seg['end_time'],
                                duration_seconds=story_seg['duration'],
                                trim_type=candidate['trim_type'],
                                confidence=float(candidate['confidence']),
                                reason=candidate['reason'],
                                text_preview=text_preview[:200] + "...",
                                impact_on_flow=candidate['impact_on_flow']
                            ))
                        
            except Exception as e:
                print(f"Error analyzing batch {i//batch_size + 1}: {e}")
                continue
        
        # Sort by confidence (safest first) then by duration (biggest impact)
        trim_candidates.sort(key=lambda x: (-x.confidence, -x.duration_seconds))
        
        return trim_candidates
    
    def find_redundant_phrases(self, transcript_segments: List[Dict]) -> List[TrimCandidate]:
        """Find specific redundant sentences and phrases within segments"""
        
        # Look for repeated phrases and filler content
        redundant_candidates = []
        
        # Common filler patterns to look for
        filler_patterns = [
            r"you know[,\s]",
            r"I mean[,\s]",
            r"sort of[,\s]",
            r"kind of[,\s]",
            r"basically[,\s]",
            r"essentially[,\s]",
            r"obviously[,\s]",
            r"clearly[,\s]",
            r"and then[,\s].*and then[,\s]",  # repeated "and then"
            r"so[,\s].*so[,\s].*so[,\s]",     # triple "so"
        ]
        
        for i, segment in enumerate(transcript_segments):
            text = segment['text']
            
            # Check for filler patterns
            for pattern in filler_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                match_count = len(list(re.finditer(pattern, text, re.IGNORECASE)))
                
                if match_count >= 2:  # Multiple instances of filler
                    redundant_candidates.append(TrimCandidate(
                        segment_id=f"Filler_{i}",
                        start_time=segment['timestamp'],
                        end_time=segment['timestamp'],  # Same segment
                        duration_seconds=match_count * 2,  # Estimated time savings
                        trim_type="repetitive",
                        confidence=0.9,  # High confidence for filler removal
                        reason=f"Multiple instances of filler phrase: {pattern}",
                        text_preview=text[:150] + "...",
                        impact_on_flow="minimal"
                    ))
        
        return redundant_candidates

def main():
    parser = argparse.ArgumentParser(description="Analyze transcript for potential content to trim")
    parser.add_argument("transcript_file", help="Path to raw transcript file")
    parser.add_argument("--segments-csv", help="Path to story segments CSV file")
    parser.add_argument("--target-minutes", type=int, default=30, help="Target minutes to trim (default: 30)")
    parser.add_argument("--export-csv", help="Export trim candidates to CSV")
    parser.add_argument("--google-api-key", help="Google API key for AI analysis")
    
    args = parser.parse_args()
    
    transcript_path = Path(args.transcript_file)
    if not transcript_path.exists():
        print(f"Transcript file not found: {transcript_path}")
        return
    
    # Get API key
    api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY found. AI-powered analysis will be skipped.")
        return
    
    print(f"Analyzing transcript for ~{args.target_minutes} minutes of potential trims...")
    
    # Load transcript
    transcript_segments = parse_raw_transcript(transcript_path)
    print(f"Loaded {len(transcript_segments)} transcript segments")
    
    # Load story segments if provided
    story_segments = []
    if args.segments_csv:
        segments_csv = Path(args.segments_csv)
        if segments_csv.exists():
            story_segments = load_story_segments(segments_csv)
            print(f"Loaded {len(story_segments)} story segments")
        else:
            print(f"Warning: Segments CSV not found: {segments_csv}")
    
    # Analyze for trim candidates
    analyzer = TrimAnalyzer(api_key)
    
    trim_candidates = []
    
    # AI-powered segment analysis
    if story_segments:
        print("\nRunning AI analysis on story segments...")
        segment_candidates = analyzer.analyze_segments_for_trimming(
            transcript_segments, story_segments, args.target_minutes
        )
        trim_candidates.extend(segment_candidates)
    
    # Pattern-based redundant phrase detection
    print("\nAnalyzing for redundant phrases and filler...")
    phrase_candidates = analyzer.find_redundant_phrases(transcript_segments)
    trim_candidates.extend(phrase_candidates)
    
    # Sort all candidates by confidence and impact
    trim_candidates.sort(key=lambda x: (-x.confidence, -x.duration_seconds))
    
    # Calculate totals
    total_potential_savings = sum(c.duration_seconds for c in trim_candidates)
    high_confidence_savings = sum(c.duration_seconds for c in trim_candidates if c.confidence >= 0.8)
    
    print(f"\n=== TRIM ANALYSIS RESULTS ===")
    print(f"Total potential time savings: {total_potential_savings // 60:.1f} minutes ({total_potential_savings} seconds)")
    print(f"High-confidence savings (≥0.8): {high_confidence_savings // 60:.1f} minutes")
    print(f"Target: {args.target_minutes} minutes")
    
    # Show top candidates
    print(f"\n=== TOP TRIM CANDIDATES ===")
    cumulative_time = 0
    for i, candidate in enumerate(trim_candidates[:20], 1):
        cumulative_time += candidate.duration_seconds
        status = "✅" if cumulative_time <= args.target_minutes * 60 else "⚠️"
        
        print(f"\n{i}. {candidate.segment_id} - {candidate.trim_type.upper()} {status}")
        print(f"   Time: {candidate.start_time} - {candidate.end_time} ({candidate.duration_seconds}s)")
        print(f"   Confidence: {candidate.confidence:.1f} | Impact: {candidate.impact_on_flow}")
        print(f"   Reason: {candidate.reason}")
        print(f"   Preview: {candidate.text_preview}")
        print(f"   Running total: {cumulative_time // 60:.1f} min")
        
        if cumulative_time >= args.target_minutes * 60:
            print(f"\n🎯 TARGET REACHED: {cumulative_time // 60:.1f} minutes with {i} removals")
            break
    
    # Export to CSV if requested
    if args.export_csv:
        output_path = Path(args.export_csv)
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['segment_id', 'start_time', 'end_time', 'duration_seconds', 
                         'trim_type', 'confidence', 'reason', 'impact_on_flow', 'text_preview']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for candidate in trim_candidates:
                writer.writerow({
                    'segment_id': candidate.segment_id,
                    'start_time': candidate.start_time,
                    'end_time': candidate.end_time,
                    'duration_seconds': candidate.duration_seconds,
                    'trim_type': candidate.trim_type,
                    'confidence': candidate.confidence,
                    'reason': candidate.reason,
                    'impact_on_flow': candidate.impact_on_flow,
                    'text_preview': candidate.text_preview
                })
        
        print(f"\n📊 Trim candidates exported to: {output_path}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Found {len(trim_candidates)} potential trim candidates")
    print(f"High-confidence options (≥0.8): {len([c for c in trim_candidates if c.confidence >= 0.8])}")
    print(f"Redundant segments: {len([c for c in trim_candidates if c.trim_type == 'redundant'])}")
    print(f"Tangential content: {len([c for c in trim_candidates if c.trim_type == 'tangent'])}")
    print(f"Non-essential sections: {len([c for c in trim_candidates if c.trim_type == 'non-essential'])}")
    print(f"Repetitive content: {len([c for c in trim_candidates if c.trim_type == 'repetitive'])}")

if __name__ == "__main__":
    main()