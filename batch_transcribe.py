#!/usr/bin/env python3
"""
Batch transcribe all audio files in a directory
"""

import os
import sys
from pathlib import Path
import subprocess
import glob

# Supported audio formats
AUDIO_EXTENSIONS = ['*.mp3', '*.m4a', '*.wav', '*.flac', '*.aac', '*.ogg', '*.mp4']

def sanitize_filename(filename):
    """Convert audio filename to a safe output filename"""
    # Remove the extension and replace problematic characters
    name = Path(filename).stem
    # Replace spaces and commas with underscores, remove other special chars
    safe_name = name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    return safe_name.lower()

def find_audio_files(directory):
    """Find all supported audio files in directory"""
    audio_files = []
    for extension in AUDIO_EXTENSIONS:
        audio_files.extend(glob.glob(os.path.join(directory, "**", extension), recursive=True))
    return audio_files

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_transcribe.py <directory_with_audio_files>")
        print(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        sys.exit(1)
    
    audio_directory = sys.argv[1]
    
    # Find all audio files
    audio_files = find_audio_files(audio_directory)
    
    if not audio_files:
        print(f"No audio files found in {audio_directory}")
        print(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files:
        print(f"  - {f}")
    
    # Create transcription directory if it doesn't exist
    os.makedirs("transcription", exist_ok=True)
    
    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n=== Processing file {i}/{len(audio_files)}: {os.path.basename(audio_file)} ===")
        
        # Generate output filename
        safe_name = sanitize_filename(audio_file)
        output_file = f"transcription/{safe_name}.md"
        
        # Run transcription
        cmd = [
            sys.executable, "transcribe.py",
            audio_file,
            output_file
        ]
        
        try:
            # Activate virtual environment and run the command
            env = os.environ.copy()
            result = subprocess.run(
                ["source", "myenv/bin/activate", "&&"] + cmd,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully transcribed: {output_file}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
            else:
                print(f"❌ Error transcribing {audio_file}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
        
        except Exception as e:
            print(f"❌ Exception while processing {audio_file}: {e}")
    
    print(f"\n🎉 Batch transcription complete! Check the transcription/ directory for outputs.")

if __name__ == "__main__":
    main()