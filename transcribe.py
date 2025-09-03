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
from typing import List, Tuple, Optional
import assemblyai as aai
from google import generativeai
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google.genai not available. Google Search integration will be limited.")
import csv
try:
    from pydub import AudioSegment
except ImportError:
    print("Warning: pydub not available. Audio enhancement will be skipped.")
    AudioSegment = None
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


@dataclass
class FactClaim:
    """A factual claim extracted from transcript"""
    claim: str
    start_timestamp: str
    end_timestamp: str
    speaker: str
    context: str
    utterance_indices: List[int]  # indices in original utterance list
    confidence_score: float = 0.0
    verdict: str = "UNVERIFIED"
    sources: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


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
        self.model = generativeai.GenerativeModel("gemini-2.5-flash")

    async def enhance_chunks(self, chunks: List[Tuple[str, Optional[io.BytesIO]]]) -> List[str]:
        """Enhance multiple transcript chunks concurrently"""
        print(f"Enhancing {len(chunks)} chunks...")
        
        # Try decreasing if you get quota errors
        semaphore = asyncio.Semaphore(15)  # Tier 1: 60 RPM for Pro model
        
        async def process_chunk(i: int, chunk: Tuple[str, io.BytesIO]) -> str:
            # Removed delay
            async with semaphore:
                text, audio = chunk
                if audio is not None:
                    audio.seek(0)
                    response = await self.model.generate_content_async(
                        [self.PROMPT, text, {"mime_type": "audio/mp3", "data": audio.read()}]
                    )
                else:
                    # Text-only enhancement
                    response = await self.model.generate_content_async(
                        [self.PROMPT, text]
                    )
                print(f"Completed chunk {i+1}/{len(chunks)}")
                return response.text

        # Create tasks for all chunks and run them concurrently
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        return results


class FactChecker:
    """Handles extracting and verifying factual claims from transcripts"""
    
    EXTRACTION_PROMPT = """You are an expert fact-checker. Your task is to identify verifiable factual claims from this transcript that could potentially be incorrect and worth re-recording if wrong.

Focus on extracting:
- Specific dates, years, numbers, statistics
- Names of people, places, companies, products
- Historical events and their details
- Technical specifications or measurements
- Claims about cause and effect relationships

Do NOT extract:
- Opinions, feelings, or subjective statements
- Future predictions or speculation
- General statements without specific details
- Common knowledge that's unlikely to be wrong

IMPORTANT: When extracting claims, ensure they include enough context to be meaningful when fact-checked in isolation. If a claim like "It involves 11,000 additions" would be meaningless without context, rephrase it as "Apollo moon landing calculations involved 11,000 additions".

For each factual claim you identify, respond with a JSON object containing:
- "claim": The factual statement WITH sufficient context to be independently verifiable
- "start_utterance": Index of first utterance containing this claim (0-based)
- "end_utterance": Index of last utterance containing this claim (0-based)
- "context": Brief description of the conversation topic (max 50 chars)

Respond with a JSON array of these objects. If no verifiable claims are found, return an empty array [].

Here are the utterances with their indices:"""

    def __init__(self, api_key: str):
        generativeai.configure(api_key=api_key)
        
        # Try to initialize Google Search capability
        if GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=api_key)
                self.grounding_tool = genai_types.Tool(
                    google_search=genai_types.GoogleSearch()
                )
                self.search_config = genai_types.GenerateContentConfig(
                    tools=[self.grounding_tool]
                )
                self.search_enabled = True
                print("Google Search integration enabled!")
            except Exception as e:
                print(f"Warning: Could not initialize Google Search: {e}")
                self.search_enabled = False
        else:
            self.search_enabled = False
            
        # Use Pro model for better accuracy in fact-checking
        self.extraction_model = generativeai.GenerativeModel("gemini-2.5-pro")

    async def extract_claims(self, utterances: List[Utterance]) -> List[FactClaim]:
        """Extract factual claims from utterances"""
        def chunk_utterances(utterances: List[Utterance], max_utterances: int = 50) -> List[List[Utterance]]:
            """Split utterances into manageable chunks for processing"""
            chunks = []
            for i in range(0, len(utterances), max_utterances):
                chunks.append(utterances[i:i + max_utterances])
            return chunks

        chunks = chunk_utterances(utterances)
        print(f"Extracting facts from {len(chunks)} chunks...")
        
        all_claims = []
        chunk_offset = 0
        
        for chunk_idx, chunk in enumerate(chunks):
            # Format utterances with indices for the prompt
            formatted_utterances = []
            for i, utterance in enumerate(chunk):
                global_idx = chunk_offset + i
                formatted_utterances.append(f"[{global_idx}] Speaker {utterance.speaker} ({utterance.timestamp}): {utterance.text}")
            
            prompt = self.EXTRACTION_PROMPT + "\n\n" + "\n".join(formatted_utterances)
            
            try:
                response = await self.extraction_model.generate_content_async(prompt)
                
                # Parse JSON response
                import re
                json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
                if json_match:
                    claims_data = json.loads(json_match.group())
                    
                    # Convert to FactClaim objects
                    for claim_data in claims_data:
                        start_idx = chunk_offset + claim_data["start_utterance"]
                        end_idx = chunk_offset + claim_data["end_utterance"]
                        
                        # Ensure indices are valid
                        if start_idx >= len(utterances) or end_idx >= len(utterances):
                            continue
                            
                        claim = FactClaim(
                            claim=claim_data["claim"],
                            start_timestamp=utterances[start_idx].timestamp,
                            end_timestamp=utterances[end_idx].timestamp,
                            speaker=utterances[start_idx].speaker,
                            context=claim_data["context"],
                            utterance_indices=list(range(start_idx, end_idx + 1))
                        )
                        all_claims.append(claim)
                        
                print(f"Extracted {len(claims_data) if 'claims_data' in locals() else 0} claims from chunk {chunk_idx + 1}")
                        
            except Exception as e:
                print(f"Error processing chunk {chunk_idx + 1}: {e}")
                continue
                
            chunk_offset += len(chunk)
            
        print(f"Total claims extracted: {len(all_claims)}")
        return all_claims

    def requires_web_search(self, claim_text: str) -> bool:
        """Determine if a claim requires Google Search or can use knowledge-based verification"""
        claim_lower = claim_text.lower()
        
        # Claims that definitely need current web search
        search_indicators = [
            # Financial/market data (changes constantly)
            'market cap', 'stock market', 'trillion', 'billion', 'million',
            'revenue', 'profit', 'stock price', 'market value', 'valuation',
            
            # Current statistics and numbers
            'employees', 'users', 'customers', 'as of', 'current', 'today',
            'now', '2024', '2025', 'recent', 'latest',
            
            # Production/manufacturing numbers
            'per day', 'per year', 'per second', 'production rate',
            
            # Recent events or quotes
            'recently said', 'announced', 'reported', 'according to',
            
            # Specific measurements that could have changed
            'square foot', 'square feet', 'times faster', 'speed',
        ]
        
        # Claims that are generally stable (use knowledge-based)
        knowledge_indicators = [
            # Historical facts (unlikely to change)
            'born in', 'founded in', 'established', 'created in',
            'invented in', 'built in', 'occurred in',
            
            # Company/person names and relationships
            'founder of', 'ceo of', 'president of', 'owned by',
            
            # Well-established historical events
            'world war', 'civil war', 'depression', 'renaissance',
            
            # Scientific/technical constants
            'speed of light', 'laws of physics', 'chemical formula',
            
            # Geographic facts
            'capital of', 'located in', 'borders',
        ]
        
        # Check for search indicators first (higher priority)
        for indicator in search_indicators:
            if indicator in claim_lower:
                return True
                
        # Check for knowledge indicators
        for indicator in knowledge_indicators:
            if indicator in claim_lower:
                return False
        
        # Default: use search for uncertain cases
        # This errs on the side of accuracy
        return True

    def export_claims_csv(self, claims: List[FactClaim], output_path: Path):
        """Export claims to CSV format"""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['claim', 'start_timestamp', 'end_timestamp', 'speaker', 
                         'fact_check_confidence', 'verdict', 'sources', 'notes', 'context']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for claim in claims:
                writer.writerow({
                    'claim': claim.claim,
                    'start_timestamp': claim.start_timestamp,
                    'end_timestamp': claim.end_timestamp,
                    'speaker': claim.speaker,
                    'fact_check_confidence': claim.confidence_score,
                    'verdict': claim.verdict,
                    'sources': ';'.join(claim.sources),
                    'notes': claim.notes,
                    'context': claim.context
                })
        print(f"Claims exported to: {output_path}")

    async def verify_claims(self, claims: List[FactClaim]) -> List[FactClaim]:
        """Verify extracted claims using Gemini Google Search"""
        VERIFICATION_PROMPT = """Fact-check this claim using web search: "{claim}"

Based on the search results, provide your assessment as JSON:
{{
    "verdict": "TRUE|FALSE|UNCERTAIN|UNVERIFIABLE",
    "confidence": 0.85,
    "notes": "Brief explanation (max 100 chars)",
    "key_sources": ["url1", "url2"]
}}

Confidence scoring:
- 0.8-1.0: Strong evidence from authoritative sources
- 0.4-0.8: Some evidence but with caveats  
- 0.0-0.4: Weak/conflicting evidence or potentially incorrect"""

        print(f"Verifying {len(claims)} claims using Google Search...")
        
        # Process claims in batches to avoid rate limits
        # Tier 1: 60 RPM for Pro model (1 per second sustained)
        batch_size = 10  # Reasonable batch size for Tier 1 
        semaphore = asyncio.Semaphore(batch_size)
        
        async def verify_single_claim(claim: FactClaim) -> FactClaim:
            async with semaphore:
                try:
                    # Add proper delay for 2 RPM limit (30 seconds between requests)
                    await asyncio.sleep(30)
                    
                    # Determine if this claim needs Google Search
                    needs_search = self.requires_web_search(claim.claim)
                    search_method = "Google Search" if needs_search else "Knowledge-based"
                    print(f"Verifying claim with {search_method}: {claim.claim[:50]}...")
                    
                    if self.search_enabled and needs_search:
                        # Use Google Search for claims that need current data
                        search_prompt = f"""Fact-check this claim using web search: "{claim.claim}"

Context from transcript: {claim.context}

Search for current, authoritative information about this claim in its given context and provide assessment as JSON:
{{
    "verdict": "TRUE|FALSE|UNCERTAIN|UNVERIFIABLE",
    "confidence": 0.85,
    "notes": "Brief explanation with current data (max 150 chars)"
}}

Confidence scoring:
- 0.8-1.0: Strong web evidence from authoritative sources
- 0.4-0.8: Some web evidence but conflicting or uncertain  
- 0.0-0.4: Web sources contradict the claim or show it's incorrect

Focus on finding the most current, accurate information."""
                        
                        # Use Google Search integration
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.client.models.generate_content(
                                model="gemini-2.5-pro",
                                contents=search_prompt,
                                config=self.search_config
                            )
                        )
                        
                        # Extract sources from response
                        sources = []
                        if hasattr(response, 'candidates') and response.candidates:
                            for candidate in response.candidates:
                                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                                    metadata = candidate.grounding_metadata
                                    
                                    # Priority 1: Extract URLs from grounding chunks (structured data)
                                    if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                                        for chunk in metadata.grounding_chunks:
                                            if hasattr(chunk, 'web') and chunk.web and hasattr(chunk.web, 'uri'):
                                                title = getattr(chunk.web, 'title', 'Unknown')
                                                sources.append(f"{title}: {chunk.web.uri}")
                                    
                                    # Priority 2: Extract URLs from search entry point HTML (fallback)
                                    elif hasattr(metadata, 'search_entry_point') and metadata.search_entry_point:
                                        entry_point = metadata.search_entry_point
                                        if hasattr(entry_point, 'rendered_content') and entry_point.rendered_content:
                                            # Parse HTML to extract href URLs and link text
                                            import re
                                            html_content = entry_point.rendered_content
                                            # Find all <a class="chip" href="..." >text</a> patterns
                                            link_pattern = r'<a class="chip" href="([^"]+)">([^<]+)</a>'
                                            matches = re.findall(link_pattern, html_content)
                                            for url, text in matches:
                                                sources.append(f"{text}: {url}")
                                    
                                    # Priority 3: older API format (final fallback)
                                    elif hasattr(metadata, 'grounding_attributions') and metadata.grounding_attributions:
                                        for attr in metadata.grounding_attributions:
                                            if hasattr(attr, 'web') and attr.web and hasattr(attr.web, 'uri'):
                                                sources.append(attr.web.uri)
                        
                        response_text = response.text
                        source_label = f"Google Search ({len(sources)} sources)" if sources else "Google Search"
                        
                    else:
                        # Use knowledge-based fact-checking for stable/historical claims
                        prompt = f"""Fact-check this claim based on your knowledge: "{claim.claim}"

Provide assessment as JSON:
{{
    "verdict": "TRUE|FALSE|UNCERTAIN|UNVERIFIABLE",
    "confidence": 0.85,
    "notes": "Brief explanation (max 100 chars)"
}}"""
                        
                        response = await self.extraction_model.generate_content_async(prompt)
                        response_text = response.text
                        sources = []
                        source_label = "Gemini AI knowledge base"
                    
                    # Parse the JSON assessment
                    import re
                    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                    if json_match:
                        try:
                            assessment = json.loads(json_match.group())
                            claim.verdict = assessment.get("verdict", "UNVERIFIABLE")
                            claim.confidence_score = float(assessment.get("confidence", 0.5))
                            claim.notes = assessment.get("notes", "Assessment completed")
                            claim.sources = sources[:3] if sources else [source_label]
                        except json.JSONDecodeError:
                            # Fallback parsing
                            claim.confidence_score = 0.5
                            claim.verdict = "UNCERTAIN"
                            claim.notes = "Could not parse verification response"
                            claim.sources = sources[:3] if sources else [source_label]
                    else:
                        # Fallback if no JSON found
                        claim.confidence_score = 0.5
                        claim.verdict = "UNCERTAIN" 
                        claim.notes = "Verification completed but format unclear"
                        claim.sources = sources[:3] if sources else [source_label]
                    
                    return claim
                    
                except Exception as e:
                    print(f"Error verifying claim '{claim.claim[:50]}...': {e}")
                    claim.confidence_score = 0.0
                    claim.verdict = "UNVERIFIABLE"
                    claim.notes = f"Error: {str(e)[:50]}"
                    return claim
        
        # Verify all claims concurrently
        verified_claims = await asyncio.gather(*[verify_single_claim(claim) for claim in claims])
        
        # Sort by confidence score (lowest first - most concerning claims first)
        verified_claims.sort(key=lambda x: x.confidence_score)
        
        return verified_claims


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


def prepare_text_only_chunks(utterances: List[Utterance]) -> List[Tuple[str, Optional[io.BytesIO]]]:
    """Prepare text-only chunks when audio processing is not available"""
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
    print(f"Preparing {len(chunks)} text-only segments...")
    
    # Process each chunk (text-only)
    prepared = []
    for chunk in chunks:
        prepared.append((format_chunk(chunk), None))
        
    return prepared


def prepare_audio_chunks(audio_path: Path, utterances: List[Utterance]) -> List[Tuple[str, Optional[io.BytesIO]]]:
    """Prepare audio chunks and their corresponding text"""
    if AudioSegment is None:
        print("Audio enhancement not available, using text-only enhancement.")
        return prepare_text_only_chunks(utterances)
        
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


async def main():
    parser = argparse.ArgumentParser(description="Create enhanced, readable transcripts from audio files")
    parser.add_argument("audio_file", help="Audio file to transcribe")
    parser.add_argument("output_file", help="Where to save the enhanced transcript")
    parser.add_argument("--assemblyai-key", help="AssemblyAI API key (can also use ASSEMBLYAI_API_KEY env var)")
    parser.add_argument("--google-key", help="Google API key (can also use GOOGLE_API_KEY env var)")
    parser.add_argument("--fact-check", action="store_true", help="Enable fact-checking of transcript claims")
    parser.add_argument("--fc-output", help="Output path for fact-check CSV (defaults to <output_file>_facts.csv)")
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

        # Save AssemblyAI transcript before enhancement
        raw_transcript = "\n\n".join(
            f"Speaker {u.speaker} {u.timestamp}\n\n{u.text}" for u in utterances
        )
        raw_output_path = output_path.with_suffix(".raw.md")
        raw_output_path.write_text(raw_transcript)
        print(f"AssemblyAI transcript saved to: {raw_output_path}")

        # Enhance transcript
        if not google_key:
            print("Google API key not provided. Skipping enhancement.")
            return
        
        
        # Enhance transcript
        enhancer = Enhancer(google_key)
        chunks = prepare_audio_chunks(audio_path, utterances)
        enhanced = asyncio.run(enhancer.enhance_chunks(chunks))
        
        # Save enhanced transcript
        merged = "\n\n".join(chunk.strip() for chunk in enhanced)
        output_path.write_text(merged)
        
        print(f"\nEnhanced transcript saved to: {output_path}")
        
        # Fact-checking if requested
        if args.fact_check:
            print("\nStarting fact-checking...")
            fact_checker = FactChecker(google_key)
            
            # Extract claims from original utterances (before enhancement)
            claims = await fact_checker.extract_claims(utterances)
            
            if claims:
                # Verify the claims
                verified_claims = await fact_checker.verify_claims(claims)
                
                # Export to CSV
                fc_output_path = Path(args.fc_output) if args.fc_output else output_path.with_suffix('.facts.csv')
                fact_checker.export_claims_csv(verified_claims, fc_output_path)
                
                # Summary of concerning claims
                low_confidence = [c for c in verified_claims if c.confidence_score < 0.4]
                if low_confidence:
                    print(f"\n⚠️  Found {len(low_confidence)} low-confidence claims that may need re-recording:")
                    for claim in low_confidence[:3]:  # Show first 3
                        print(f"  • {claim.start_timestamp}: {claim.claim[:60]}...")
                    if len(low_confidence) > 3:
                        print(f"  ... and {len(low_confidence) - 3} more (see CSV for full list)")
                else:
                    print("\n✅ No low-confidence claims found!")
            else:
                print("No factual claims detected in transcript.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())