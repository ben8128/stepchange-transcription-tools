#!/usr/bin/env python3
"""
Human quality transcripts from audio files using 
AssemblyAI for transcription and Google's Gemini for enhancement.

Requirements:
- AssemblyAI API key (https://www.assemblyai.com/)
- Google API key (https://aistudio.google.com/)
- Python packages: assemblyai, google-generativeai, pydub

Usage:
python transcribe.py input.mp3 output.md
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import List, Tuple
import assemblyai as aai
from google import generativeai
from pydub import AudioSegment
import asyncio
import io


@dataclass
class Utterance:
    """A single utterance from a speaker"""
    speaker: str
    text: str
    start: int  # timestamp in ms
    end: int    # timestamp in ms

    @property
    def timestamp(self) -> str:
        """Format start time as HH:MM:SS"""
        seconds = int(self.start // 1000)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Transcriber:
    """Handles getting transcripts from AssemblyAI"""

    def __init__(self, api_key: str):
        aai.settings.api_key = api_key

    def transcribe(self, audio_path: Path) -> List[Utterance]:
        """Get transcript from AssemblyAI"""
        print("Getting transcript from AssemblyAI...")
        config = aai.TranscriptionConfig(speaker_labels=True, language_code="en")
        transcript = aai.Transcriber().transcribe(str(audio_path), config=config)
        
        return [
            Utterance(speaker=u.speaker, text=u.text, start=u.start, end=u.end)
            for u in transcript.utterances
        ]


class Enhancer:
    """Handles enhancing transcripts using Gemini"""

    PROMPT = """You are an expert transcript editor. Your task is to enhance this transcript for maximum readability while maintaining the core message.
IMPORTANT: Respond ONLY with the enhanced transcript. Do not include any explanations, headers, or phrases like "Here is the transcript."
Note: Below you'll find an auto-generated transcript that may help with speaker identification, but focus on creating your own high-quality transcript from the audio.

Think about your job as if you were transcribing an interview for a print book where the priority is the reading audience. It should just be a total pleasure to read this as a written artifact where all the flubs and repetitions and conversational artifacts and filler words and false starts are removed, where a bunch of helpful punctuation is added. It should basically read like somebody wrote it specifically for reading rather than just something somebody said extemporaneously.

Please:
1. Fix speaker attribution errors, especially at segment boundaries. Watch for incomplete thoughts that were likely from the previous speaker.

2. Optimize AGGRESSIVELY for readability over verbatim accuracy:
   * Readability is the most important thing!!
   * Remove ALL conversational artifacts (yeah, so, I mean, etc.)
   * Remove ALL filler words (um, uh, like, you know)
   * Remove false starts and self-corrections completely
   * Remove redundant phrases and hesitations
   * Convert any indirect or rambling responses into direct statements
   * Break up run-on sentences into clear, concise statements
   * Maintain natural conversation flow while prioritizing clarity and directness

3. Format the output consistently:
   * Keep the "Speaker X 00:00:00" format (no brackets, no other formatting)
   * DO NOT change the timestamps. You're only seeing a chunk of the full transcript, which is why your 0:00:00 is not the true beginning. Keep the timestamps as they are.
   * Add TWO line breaks between speaker/timestamp and the text
   * Use proper punctuation and capitalization
   * Add paragraph breaks for topic changes
   * When you add paragraph breaks between the same speaker's remarks, no need to restate the speaker attribution
   * Don't go more than four sentences without adding a paragraph break. Be liberal with your paragraph breaks.
   * Preserve distinct speaker turns

Example input:
Speaker A 00:01:15

Um, yeah, so like, I've been working on this new project at work, you know? And uh, what's really interesting is that, uh, we're seeing these amazing results with the new approach we're taking. Like, it's just, you know, it's really transforming how we do things. And then, I mean, the thing is, uh, when we showed it to the client last week, they were just, you know, completely blown away by what we achieved. Like, they couldn't even believe it was the same system they had before.

Example output:
Speaker A 00:01:15

I've been working on this new project at work, and we're seeing amazing results with our new approach. It's really transforming how we do things.

When we showed it to the client last week, they were completely blown away by what we achieved. They couldn't believe it was the same system they had before.

Enhance the following transcript, starting directly with the speaker format:"""

    def __init__(self, api_key: str):
        generativeai.configure(api_key=api_key)
        self.model = generativeai.GenerativeModel("gemini-exp-1206")
        self.rate_limit_delay = 1.0  # seconds between requests

    async def enhance_chunks(self, chunks: List[Tuple[str, io.BytesIO]]) -> List[str]:
        """Enhance multiple transcript chunks concurrently with rate limiting"""
        print(f"Enhancing {len(chunks)} chunks...")
        
        # Create a semaphore to limit concurrent requests
        # Kinda slow, but I got rate limited when I tried to do more
        semaphore = asyncio.Semaphore(3)  # Allow up to 3 concurrent requests
        
        async def process_chunk(i: int, chunk: Tuple[str, io.BytesIO]) -> str:
            text, audio = chunk
            async with semaphore:
                if i > 0:  # Don't delay first request
                    await asyncio.sleep(self.rate_limit_delay)
                audio.seek(0)
                response = await self.model.generate_content_async(
                    [self.PROMPT, text, {"mime_type": "audio/mp3", "data": audio.read()}]
                )
                print(f"Completed chunk {i+1}/{len(chunks)}")
                return response.text

        # Create tasks for all chunks and run them concurrently
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        return results


def format_chunk(utterances: List[Utterance]) -> str:
    """Format utterances into readable text with timestamps"""
    sections = []
    current_speaker = None
    current_texts = []
    
    for u in utterances:
        if current_speaker != u.speaker:
            if current_texts:
                sections.append(f"Speaker {current_speaker} {utterances[len(sections)].timestamp}\n\n{''.join(current_texts)}")
            current_speaker = u.speaker
            current_texts = []
        current_texts.append(u.text)
    
    if current_texts:
        sections.append(f"Speaker {current_speaker} {utterances[len(sections)].timestamp}\n\n{''.join(current_texts)}")
    
    return "\n\n".join(sections)


def prepare_audio_chunks(audio_path: Path, utterances: List[Utterance]) -> List[Tuple[str, io.BytesIO]]:
    """Prepare audio chunks and their corresponding text"""
    def chunk_utterances(utterances: List[Utterance], max_tokens: int = 8000) -> List[List[Utterance]]:
        chunks = []
        current = []
        text_length = 0
        
        for u in utterances:
            new_length = text_length + len(u.text)
            if current and new_length > max_tokens:
                chunks.append(current)
                current = [u]
                text_length = len(u.text)
            else:
                current.append(u)
                text_length = new_length
                
        if current:
            chunks.append(current)
        return chunks

    # Split utterances into chunks
    chunks = chunk_utterances(utterances)
    print(f"Preparing {len(chunks)} audio segments...")
    
    # Load audio once
    audio = AudioSegment.from_file(audio_path)
    
    # Process each chunk
    prepared = []
    for chunk in chunks:
        # Extract just the needed segment
        segment = audio[chunk[0].start:chunk[-1].end]
        buffer = io.BytesIO()
        # Use lower quality MP3 for faster processing
        segment.export(buffer, format="mp3", parameters=["-q:a", "9"])
        prepared.append((format_chunk(chunk), buffer))
        
    return prepared


def main():
    parser = argparse.ArgumentParser(description="Create enhanced, readable transcripts from audio files")
    parser.add_argument("audio_file", help="Audio file to transcribe")
    parser.add_argument("output_file", help="Where to save the enhanced transcript")
    parser.add_argument("--assemblyai-key", help="AssemblyAI API key (can also use ASSEMBLYAI_API_KEY env var)")
    parser.add_argument("--google-key", help="Google API key (can also use GOOGLE_API_KEY env var)")
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    output_path = Path(args.output_file)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
        
    try:
        # Get API keys from environment or command line arguments
        assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY") or args.assemblyai_key
        google_key = os.getenv("GOOGLE_API_KEY") or args.google_key
        
        if not assemblyai_key or not google_key:
            raise ValueError(
                "Please provide API keys either through environment variables "
                "(ASSEMBLYAI_API_KEY and GOOGLE_API_KEY) or command line arguments "
                "(--assemblyai-key and --google-key)"
            )
        
        # Get transcript
        transcriber = Transcriber(assemblyai_key)
        utterances = transcriber.transcribe(audio_path)
        
        # Enhance transcript
        enhancer = Enhancer(google_key)
        chunks = prepare_audio_chunks(audio_path, utterances)
        enhanced = asyncio.run(enhancer.enhance_chunks(chunks))
        
        # Save enhanced transcript
        merged = "\n\n".join(chunk.strip() for chunk in enhanced)
        output_path.write_text(merged)
        
        print(f"\nEnhanced transcript saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()