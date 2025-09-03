# Claude Code Configuration

This file contains commands and configurations for working with this transcription and fact-checking toolkit.

## Common Commands

### Setup & Dependencies
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Transcription
```bash
# Transcribe audio file
python3 transcribe_only.py path/to/audio.m4a

# Batch transcribe multiple files
python3 batch_transcribe.py path/to/audio/directory/
```

### Fact Checking
```bash
# Fact-check a transcript in chunks (recommended)
python3 factcheck_chunked.py transcription/transcript.raw.md --export-csv results.csv

# Test fact-checking on small sample
python3 factcheck_chunked.py transcription/transcript.raw.md --max-utterances 20
```

### Analysis
```bash
# Analyze speaker sections for dominance patterns
python3 analyze_speaker_sections.py transcription/transcript.raw.md --export-csv speaker_analysis.csv

# Test context extraction
python3 test_context_extraction.py

# Debug source extraction
python3 debug_source_extraction.py
```

### Testing
```bash
# Run linting and type checking
python3 -m flake8 *.py
python3 -m mypy *.py --ignore-missing-imports
```

## Environment Variables Required
- `GOOGLE_API_KEY`: For Gemini API and Google Search grounding
- `ASSEMBLYAI_API_KEY`: For audio transcription

## Key Features
- Audio transcription with speaker diarization
- Contextual fact-checking with source URLs
- Speaker dominance analysis
- Chunk-based processing for large transcripts
- CSV export for all analysis results