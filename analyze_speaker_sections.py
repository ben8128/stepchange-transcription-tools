#!/usr/bin/env python3
"""
Analyze transcript for sections where one speaker dominates for extended periods.
Identifies both long individual segments and multi-minute chunks of speaker dominance.
"""
import argparse
from pathlib import Path
import csv
from dataclasses import dataclass
from typing import List, Tuple
import statistics

@dataclass
class SegmentInfo:
    speaker: str
    start_time: str
    end_time: str 
    duration_seconds: int
    text: str
    word_count: int

@dataclass
class SpeakerDominanceChunk:
    start_time: str
    end_time: str
    duration_seconds: int
    dominant_speaker: str
    speaker_ratio: float  # percentage of time dominant speaker spoke
    segments_count: int
    total_words: int

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

def parse_raw_transcript(file_path: Path) -> List[SegmentInfo]:
    """Parse a raw transcript file into SegmentInfo objects"""
    content = file_path.read_text()
    lines = content.strip().split('\n')
    
    segments = []
    current_speaker = None
    current_time = None
    current_text = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a speaker/timestamp line (format: "A 00:00:00" or "B 00:00:04")
        if len(line.split()) == 2:
            parts = line.split()
            if len(parts[0]) == 1 and ':' in parts[1]:  # Single letter speaker + timestamp
                # Save previous segment if exists
                if current_speaker and current_text:
                    text = ' '.join(current_text).strip()
                    if text:
                        # Find the next timestamp to calculate duration
                        next_timestamp = None
                        for j in range(i + 1, len(lines)):
                            next_line = lines[j].strip()
                            if len(next_line.split()) == 2:
                                next_parts = next_line.split()
                                if len(next_parts[0]) == 1 and ':' in next_parts[1]:
                                    next_timestamp = next_parts[1]
                                    break
                        
                        start_seconds = parse_timestamp_to_seconds(current_time)
                        end_seconds = parse_timestamp_to_seconds(next_timestamp) if next_timestamp else start_seconds + 5
                        duration = end_seconds - start_seconds
                        
                        segments.append(SegmentInfo(
                            speaker=current_speaker,
                            start_time=current_time,
                            end_time=next_timestamp or current_time,
                            duration_seconds=duration,
                            text=text,
                            word_count=len(text.split())
                        ))
                
                # Start new segment
                current_speaker = parts[0]  # Just "A" or "B"
                current_time = parts[1]     # Just the timestamp
                current_text = []
                continue
        
        # Add text to current segment
        if current_speaker:
            current_text.append(line)
    
    # Don't forget the last segment
    if current_speaker and current_text:
        text = ' '.join(current_text).strip()
        if text:
            segments.append(SegmentInfo(
                speaker=current_speaker,
                start_time=current_time,
                end_time=current_time,
                duration_seconds=5,  # Estimate for last segment
                text=text,
                word_count=len(text.split())
            ))
    
    return segments

def find_long_segments(segments: List[SegmentInfo], threshold_seconds: int = 60) -> List[SegmentInfo]:
    """Find individual segments longer than threshold"""
    return [seg for seg in segments if seg.duration_seconds >= threshold_seconds]

def analyze_speaker_dominance_chunks(segments: List[SegmentInfo], chunk_duration_seconds: int = 120) -> List[SpeakerDominanceChunk]:
    """Analyze chunks of time to find periods where one speaker dominates"""
    if not segments:
        return []
    
    chunks = []
    start_time_seconds = parse_timestamp_to_seconds(segments[0].start_time)
    end_time_seconds = parse_timestamp_to_seconds(segments[-1].start_time)
    
    current_chunk_start = start_time_seconds
    
    while current_chunk_start < end_time_seconds:
        chunk_end = current_chunk_start + chunk_duration_seconds
        
        # Find segments that fall within this chunk
        chunk_segments = []
        for seg in segments:
            seg_start = parse_timestamp_to_seconds(seg.start_time)
            if current_chunk_start <= seg_start < chunk_end:
                chunk_segments.append(seg)
        
        if chunk_segments:
            # Calculate speaker time and word counts
            speaker_time = {'A': 0, 'B': 0}
            speaker_words = {'A': 0, 'B': 0}
            
            for seg in chunk_segments:
                speaker_time[seg.speaker] += seg.duration_seconds
                speaker_words[seg.speaker] += seg.word_count
            
            total_time = sum(speaker_time.values())
            total_words = sum(speaker_words.values())
            
            if total_time > 0:
                dominant_speaker = max(speaker_time.keys(), key=lambda x: speaker_time[x])
                dominance_ratio = speaker_time[dominant_speaker] / total_time
                
                # Convert back to timestamp format
                chunk_start_str = f"{current_chunk_start//3600:02d}:{(current_chunk_start%3600)//60:02d}:{current_chunk_start%60:02d}"
                chunk_end_str = f"{chunk_end//3600:02d}:{(chunk_end%3600)//60:02d}:{chunk_end%60:02d}"
                
                chunks.append(SpeakerDominanceChunk(
                    start_time=chunk_start_str,
                    end_time=chunk_end_str,
                    duration_seconds=chunk_duration_seconds,
                    dominant_speaker=dominant_speaker,
                    speaker_ratio=dominance_ratio,
                    segments_count=len(chunk_segments),
                    total_words=total_words
                ))
        
        current_chunk_start = chunk_end
    
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Analyze transcript for long single-speaker sections")
    parser.add_argument("transcript_file", help="Path to the raw transcript file")
    parser.add_argument("--segment-threshold", type=int, default=60, help="Minimum seconds for long individual segments (default: 60)")
    parser.add_argument("--chunk-duration", type=int, default=120, help="Duration of chunks to analyze for dominance in seconds (default: 120)")
    parser.add_argument("--dominance-threshold", type=float, default=0.7, help="Minimum ratio for speaker dominance (default: 0.7)")
    parser.add_argument("--export-csv", help="Export results to CSV file")
    
    args = parser.parse_args()
    
    transcript_path = Path(args.transcript_file)
    if not transcript_path.exists():
        print(f"File not found: {transcript_path}")
        return
    
    print(f"Analyzing transcript: {transcript_path}")
    
    # Parse transcript
    segments = parse_raw_transcript(transcript_path)
    print(f"Parsed {len(segments)} segments")
    
    if not segments:
        print("No segments found")
        return
    
    # Calculate basic stats
    durations = [seg.duration_seconds for seg in segments]
    word_counts = [seg.word_count for seg in segments]
    
    print(f"\n=== Basic Statistics ===")
    print(f"Average segment duration: {statistics.mean(durations):.1f} seconds")
    print(f"Median segment duration: {statistics.median(durations):.1f} seconds")
    print(f"Average words per segment: {statistics.mean(word_counts):.1f}")
    print(f"Total duration: {sum(durations)} seconds ({sum(durations)//60:.1f} minutes)")
    
    # Find long individual segments
    long_segments = find_long_segments(segments, args.segment_threshold)
    print(f"\n=== Long Individual Segments (>{args.segment_threshold}s) ===")
    print(f"Found {len(long_segments)} long segments:")
    
    for seg in sorted(long_segments, key=lambda x: x.duration_seconds, reverse=True)[:10]:
        print(f"  {seg.start_time} - Speaker {seg.speaker}: {seg.duration_seconds}s ({seg.word_count} words)")
        print(f"    \"{seg.text[:100]}...\"")
        print()
    
    # Analyze speaker dominance in chunks
    dominance_chunks = analyze_speaker_dominance_chunks(segments, args.chunk_duration)
    high_dominance = [chunk for chunk in dominance_chunks if chunk.speaker_ratio >= args.dominance_threshold]
    
    print(f"\n=== Speaker Dominance Analysis ({args.chunk_duration}s chunks, >{args.dominance_threshold*100:.0f}% dominance) ===")
    print(f"Found {len(high_dominance)} high-dominance chunks:")
    
    for chunk in sorted(high_dominance, key=lambda x: x.speaker_ratio, reverse=True)[:10]:
        print(f"  {chunk.start_time}-{chunk.end_time}: Speaker {chunk.dominant_speaker} dominates {chunk.speaker_ratio*100:.1f}%")
        print(f"    {chunk.segments_count} segments, {chunk.total_words} total words")
        print()
    
    # Export to CSV if requested
    if args.export_csv:
        with open(args.export_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write long segments
            writer.writerow(["=== LONG INDIVIDUAL SEGMENTS ==="])
            writer.writerow(["Start Time", "Speaker", "Duration (s)", "Word Count", "Text Preview"])
            for seg in sorted(long_segments, key=lambda x: x.duration_seconds, reverse=True):
                writer.writerow([seg.start_time, seg.speaker, seg.duration_seconds, seg.word_count, seg.text[:200]])
            
            writer.writerow([])  # Empty row
            
            # Write dominance chunks
            writer.writerow(["=== HIGH SPEAKER DOMINANCE CHUNKS ==="])
            writer.writerow(["Start Time", "End Time", "Dominant Speaker", "Dominance %", "Segments", "Total Words"])
            for chunk in sorted(high_dominance, key=lambda x: x.speaker_ratio, reverse=True):
                writer.writerow([chunk.start_time, chunk.end_time, chunk.dominant_speaker, 
                               f"{chunk.speaker_ratio*100:.1f}%", chunk.segments_count, chunk.total_words])
        
        print(f"\n📊 Results exported to: {args.export_csv}")

if __name__ == "__main__":
    main()