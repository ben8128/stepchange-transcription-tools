#!/usr/bin/env python3
"""
Fact-check an existing raw transcript file in chunks
"""
import asyncio
import os
import sys
import argparse
from pathlib import Path
from transcribe import Utterance, FactChecker

def parse_timestamp_to_ms(timestamp_str: str) -> int:
    """Convert HH:MM:SS timestamp to milliseconds"""
    try:
        parts = timestamp_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            return total_seconds * 1000
        else:
            # Fallback for invalid format
            return 0
    except (ValueError, AttributeError):
        return 0

def parse_raw_transcript(file_path: Path, max_utterances: int = None) -> list:
    """Parse a raw transcript file into Utterance objects"""
    content = file_path.read_text()
    lines = content.strip().split('\n')
    
    utterances = []
    current_speaker = None
    current_time = None
    current_text = []
    utterance_index = 0
    
    for line in lines:
        if max_utterances and len(utterances) >= max_utterances:
            break
            
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
                            # Convert timestamp to milliseconds from actual timestamp
                            time_ms = parse_timestamp_to_ms(current_time)
                            utterances.append(Utterance(current_speaker, text, time_ms, time_ms + 5000))
                            utterance_index += 1
                    
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
    if current_speaker and current_text and (not max_utterances or len(utterances) < max_utterances):
        text = ' '.join(current_text).strip()
        if text:
            time_ms = parse_timestamp_to_ms(current_time)
            utterances.append(Utterance(current_speaker, text, time_ms, time_ms + 5000))
    
    return utterances

def chunk_utterances(utterances: list, chunk_size: int = 50) -> list:
    """Split utterances into manageable chunks"""
    chunks = []
    for i in range(0, len(utterances), chunk_size):
        chunks.append(utterances[i:i + chunk_size])
    return chunks

async def main():
    parser = argparse.ArgumentParser(description="Fact-check an existing raw transcript file in chunks")
    parser.add_argument("transcript_file", help="Path to the raw transcript file")
    parser.add_argument("--max-utterances", type=int, help="Maximum number of utterances to process (for testing)")
    parser.add_argument("--chunk-size", type=int, default=50, help="Number of utterances per chunk (default: 50)")
    
    args = parser.parse_args()
    
    transcript_path = Path(args.transcript_file)
    if not transcript_path.exists():
        print(f"File not found: {transcript_path}")
        return
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY environment variable found")
        return
    
    print(f"Loading transcript: {transcript_path}")
    if args.max_utterances:
        print(f"Processing first {args.max_utterances} utterances only")
    
    utterances = parse_raw_transcript(transcript_path, args.max_utterances)
    print(f"Parsed {len(utterances)} utterances")
    
    # Split into chunks
    chunks = chunk_utterances(utterances, chunk_size=args.chunk_size)
    print(f"Split into {len(chunks)} chunks of ~{args.chunk_size} utterances each")
    
    fact_checker = FactChecker(api_key)
    all_verified_claims = []
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n=== Processing Chunk {i}/{len(chunks)} ===")
        print(f"Processing utterances {chunk[0].start//1000}s to {chunk[-1].start//1000}s")
        
        try:
            # Extract claims from this chunk
            claims = await fact_checker.extract_claims(chunk)
            print(f"Extracted {len(claims)} claims from chunk {i}")
            
            # Verify claims
            if claims:
                verified_claims = await fact_checker.verify_claims(claims)
                all_verified_claims.extend(verified_claims)
                print(f"Verified {len(verified_claims)} claims from chunk {i}")
                
                # Show a few sample results
                for claim in verified_claims[:3]:  # Show first 3
                    print(f"  - {claim.claim[:60]}... -> {claim.verdict} ({claim.confidence_score:.2f})")
            
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            continue
    
    # Export all results
    if all_verified_claims:
        output_path = transcript_path.stem + "_facts_chunked.csv"
        fact_checker.export_claims_csv(all_verified_claims, Path(output_path))
        print(f"\n🎉 Exported {len(all_verified_claims)} total verified claims to: {output_path}")
        
        # Show summary stats
        verdicts = [claim.verdict for claim in all_verified_claims]
        print(f"\nSummary:")
        print(f"  TRUE: {verdicts.count('TRUE')}")
        print(f"  FALSE: {verdicts.count('FALSE')}")
        print(f"  UNCERTAIN: {verdicts.count('UNCERTAIN')}")
        print(f"  UNVERIFIABLE: {verdicts.count('UNVERIFIABLE')}")
        
        # Show lowest confidence claims (most concerning)
        low_confidence = sorted([c for c in all_verified_claims if c.confidence_score < 0.7], 
                               key=lambda x: x.confidence_score)
        if low_confidence:
            print(f"\n🚨 {len(low_confidence)} claims with confidence < 0.7 (may need re-recording):")
            for claim in low_confidence[:5]:  # Show worst 5
                print(f"  {claim.confidence_score:.2f}: {claim.claim[:80]}...")

if __name__ == "__main__":
    asyncio.run(main())