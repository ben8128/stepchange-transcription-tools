#!/usr/bin/env python3
"""
Simple transcription-only script
"""
import argparse
from pathlib import Path
import os
from transcribe import Transcriber

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to raw transcript only")
    parser.add_argument("audio_file", help="Audio file to transcribe")
    parser.add_argument("output_file", help="Where to save the raw transcript")
    parser.add_argument("--assemblyai-key", help="AssemblyAI API key (can also use ASSEMBLYAI_API_KEY env var)")
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    output_path = Path(args.output_file)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
        
    try:
        # Get API key from environment or command line arguments
        assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY") or args.assemblyai_key
        
        if not assemblyai_key:
            raise ValueError(
                "Please provide AssemblyAI API key either through environment variable "
                "(ASSEMBLYAI_API_KEY) or command line argument (--assemblyai-key)"
            )
        
        # Get transcript only
        transcriber = Transcriber(assemblyai_key)
        utterances = transcriber.transcribe(audio_path)

        # Save raw transcript
        raw_transcript = "\n\n".join(
            f"{u.speaker} {u.timestamp}\n\n{u.text}" for u in utterances
        )
        output_path.write_text(raw_transcript)
        print(f"Raw transcript saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()