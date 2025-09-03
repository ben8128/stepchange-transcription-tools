#!/usr/bin/env python3
"""
Full transcription pipeline: raw transcription + enhancement + optional fact-checking.
Combines transcribe_only.py and enhance_transcript.py into one workflow.
"""
import argparse
from pathlib import Path
import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def main():
    parser = argparse.ArgumentParser(description="Full transcription pipeline with enhancement")
    parser.add_argument("audio_file", help="Audio file to transcribe")
    parser.add_argument("--output-dir", default="transcription", help="Output directory (default: transcription)")
    parser.add_argument("--fact-check", action="store_true", help="Also run fact-checking on the transcript")
    parser.add_argument("--no-enhance", action="store_true", help="Skip enhancement step")
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames
    base_name = audio_path.stem
    raw_output = output_dir / f"{base_name}.raw.md"
    enhanced_output = output_dir / f"{base_name}.enhanced.md"
    facts_output = output_dir / f"{base_name}.facts.csv"
    
    print(f"""
📋 TRANSCRIPTION PIPELINE
========================
Audio file: {audio_path}
Output directory: {output_dir}

Steps:
1. Raw transcription → {raw_output}
{"2. Enhancement → " + str(enhanced_output) if not args.no_enhance else "2. Enhancement: SKIPPED"}
{"3. Fact-checking → " + str(facts_output) if args.fact_check else "3. Fact-checking: SKIPPED"}
""")
    
    # Step 1: Raw transcription
    if not run_command(
        f'python3 transcribe_only.py "{audio_path}" "{raw_output}"',
        "Step 1: Transcribing audio to raw transcript"
    ):
        print("❌ Transcription failed")
        sys.exit(1)
    
    print(f"✅ Raw transcript saved to: {raw_output}")
    
    # Step 2: Enhancement (optional)
    if not args.no_enhance:
        if not run_command(
            f'python3 enhance_transcript.py "{raw_output}" --output "{enhanced_output}"',
            "Step 2: Enhancing transcript for readability"
        ):
            print("⚠️  Enhancement failed, but raw transcript is available")
        else:
            print(f"✅ Enhanced transcript saved to: {enhanced_output}")
    
    # Step 3: Fact-checking (optional)
    if args.fact_check:
        # Use smaller chunks for fact-checking to avoid rate limits
        if not run_command(
            f'python3 factcheck_chunked.py "{raw_output}" --chunk-size 30',
            "Step 3: Fact-checking transcript claims"
        ):
            print("⚠️  Fact-checking failed or was incomplete")
        else:
            # The script automatically generates a CSV with the naming convention
            expected_facts = raw_output.parent / f"{raw_output.stem}_facts_chunked.csv"
            if expected_facts.exists():
                # Move to our preferred location
                expected_facts.rename(facts_output)
                print(f"✅ Fact-check results saved to: {facts_output}")
    
    print(f"""
✨ PIPELINE COMPLETE
====================
Raw transcript: {raw_output}
{"Enhanced transcript: " + str(enhanced_output) if not args.no_enhance and enhanced_output.exists() else ""}
{"Fact-check results: " + str(facts_output) if args.fact_check and facts_output.exists() else ""}

You can now:
- Review the raw transcript for accuracy
- Share the enhanced transcript for better readability
- Check the fact verification results if generated
""")

if __name__ == "__main__":
    main()