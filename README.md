# Podcast Transcription & Fact-Checking Toolkit

A comprehensive toolkit for transcribing podcast audio, fact-checking content with AI-powered research, and analyzing speaker patterns. Built with AssemblyAI for transcription and Google Gemini 2.5 Pro for contextual fact-checking with web search integration.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- Audio files in supported formats (M4A, MP3, WAV, etc.)

### 2. Installation
```bash
git clone <your-repo-url>
cd transcribe

# Recommended: Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. API Keys Setup
You'll need two API keys:

**Google API Key** (for fact-checking):
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. **IMPORTANT**: Enable billing for higher rate limits
   - Free tier: 2 requests/minute (very limited for fact-checking)
   - Paid tier: Much higher limits, enables full processing
   - Go to [Google Cloud Console](https://console.cloud.google.com/) to set up billing
4. Set environment variable: `export GOOGLE_API_KEY="your_key_here"`

**AssemblyAI API Key** (for transcription):
1. Sign up at [AssemblyAI](https://www.assemblyai.com/)
2. Get your API key from the dashboard
3. Set environment variable: `export ASSEMBLYAI_API_KEY="your_key_here"`

### 4. Basic Usage
```bash
# Transcribe an audio file
python3 transcribe_only.py path/to/your/podcast.m4a

# Fact-check the transcript
python3 factcheck_chunked.py transcription/your_podcast.raw.md --export-csv results.csv

# Analyze speaker patterns
python3 analyze_speaker_sections.py transcription/your_podcast.raw.md --export-csv speaker_analysis.csv
```

## 📋 Core Workflows

### 🎤 Transcription
Convert audio to text with speaker diarization:
```bash
# Single file transcription (raw output)
python3 transcribe_only.py audio/podcast.m4a transcript.raw.md

# Full pipeline: transcription + enhancement
python3 transcribe_full.py audio/podcast.m4a

# Batch transcription
python3 batch_transcribe.py audio/directory/

# Output: Creates .raw.md files with speaker-labeled transcript
```

### ✨ Enhancement
Create clean, readable versions of raw transcripts:
```bash
# Enhance a raw transcript for better readability
python3 enhance_transcript.py transcript.raw.md

# Output: Creates .enhanced.md with:
# - Grammar and punctuation fixes
# - Removed filler words
# - Natural paragraph breaks
# - Preserved speaker voice and authenticity
```

### 📊 Segmentation & Analysis
Analyze transcript patterns and speaker dynamics:
```bash
# Find long monologues and speaker dominance patterns
python3 analyze_speaker_sections.py transcript.raw.md --export-csv analysis.csv

# Customize thresholds
python3 analyze_speaker_sections.py transcript.raw.md \
  --segment-threshold 90 \
  --chunk-duration 180 \
  --dominance-threshold 0.8
```

### 🔍 Fact Checking
AI-powered fact verification with web search:
```bash
# Full fact-checking with contextual claims
python3 factcheck_chunked.py transcript.raw.md --export-csv verified_facts.csv

# Test on smaller sample first
python3 factcheck_chunked.py transcript.raw.md --max-utterances 50

# Adjust chunk size for rate limiting
python3 factcheck_chunked.py transcript.raw.md --chunk-size 25
```

## 📁 File Structure

```
transcribe/
├── README.md                          # This file
├── CLAUDE.md                          # Claude Code configuration
├── requirements.txt                   # Python dependencies
├── transcribe.py                      # Core transcription & fact-checking classes
├── transcribe_only.py                 # Audio → transcript conversion
├── batch_transcribe.py                # Bulk transcription processing
├── factcheck_chunked.py              # Chunked fact-checking for large files
├── analyze_speaker_sections.py       # Speaker pattern analysis
└── transcription/                     # Output directory
    ├── your_podcast.raw.md           # Raw transcript with timestamps
    ├── your_podcast_facts.csv        # Fact-check results
    └── your_podcast_analysis.csv     # Speaker analysis results
```

## 🎯 Key Features

### Transcription
- **Speaker Diarization**: Automatically identifies and labels speakers (A/B)
- **Timestamp Precision**: Accurate timing for each segment
- **Format Support**: M4A, MP3, WAV, and other common audio formats
- **Batch Processing**: Handle multiple files at once

### Fact Checking
- **Contextual Claims**: Extracts meaningful, verifiable claims from conversation context
- **Web Search Integration**: Uses Google Search for authoritative sources
- **Source Attribution**: Includes URLs and titles for verification
- **Confidence Scoring**: AI-powered confidence ratings for each claim
- **Chunked Processing**: Handles long transcripts efficiently

### Speaker Analysis
- **Dominance Patterns**: Identifies sections where one speaker dominates
- **Long Monologues**: Finds extended single-speaker segments
- **Balance Metrics**: Overall word count and time distribution
- **Intervention Points**: Suggests where to add speaker interjections

## 🔧 Advanced Configuration

### Fact Checking Parameters
```bash
# Adjust for different content types
python3 factcheck_chunked.py transcript.raw.md \
  --chunk-size 30 \           # Smaller chunks for dense content
  --max-utterances 100        # Limit for testing
```

### Speaker Analysis Options
```bash
python3 analyze_speaker_sections.py transcript.raw.md \
  --segment-threshold 120 \    # Longer threshold for monologues
  --chunk-duration 300 \       # 5-minute dominance windows
  --dominance-threshold 0.75   # 75% dominance requirement
```

## 📊 Output Formats

### Transcript (.raw.md)
```
A 00:00:00
Welcome to the podcast about data centers.

B 00:00:15
Thanks for having me! Let's dive into the infrastructure.
```

### Fact Check Results (.csv)
| Claim | Verdict | Fact Check Confidence | Sources | Context |
|-------|---------|----------------------|---------|---------|
| Data centers consume 1% of global electricity | TRUE | 0.85 | IEA Report: https://... | Discussion about energy usage |

**Note:** `Fact Check Confidence` (0.0-1.0) indicates the AI's confidence in its verdict based on source quality and relevance.

### Speaker Analysis (.csv)
| Type | Start Time | Speaker | Duration | Details |
|------|------------|---------|----------|---------|
| Long Segment | 00:15:30 | A | 180s | Extended explanation |
| Dominance Chunk | 00:20:00-00:22:00 | B | 85% | Technical discussion |

## 🐛 Troubleshooting

### Common Issues

**"No API key found"**
```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Set your environment variables
export GOOGLE_API_KEY="your_google_api_key"
export ASSEMBLYAI_API_KEY="your_assemblyai_key"
```

**"Rate limit exceeded"**
```bash
# Use smaller chunk sizes
python3 factcheck_chunked.py transcript.raw.md --chunk-size 25
```
**Note:** If you consistently hit rate limits:
- You're likely on the free tier (2 requests/minute)
- Enable billing in [Google Cloud Console](https://console.cloud.google.com/)
- Or process smaller samples: `--max-utterances 50`

**"No sources found in fact-checking"**
- Verify your Google API key has access to search
- Check that claims are being extracted properly with `debug_source_extraction.py`

### Debug Tools
```bash
# Test context extraction
python3 test_context_extraction.py

# Debug source URL extraction
python3 debug_source_extraction.py

# Test small fact-checking samples
python3 factcheck_chunked.py transcript.raw.md --max-utterances 5
```

## 🤝 Contributing

This toolkit was built for podcast analysis and fact-checking workflows. Feel free to:
- Add support for additional audio formats
- Improve claim extraction algorithms  
- Add new analysis metrics
- Enhance the speaker segmentation logic

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

**Built by [Stepchange](https://stepchange.show)** - A podcast exploring the step changes that shape our world.

---

**Need help?** Check the debug tools in the repo or review the example outputs in the `transcription/` directory.