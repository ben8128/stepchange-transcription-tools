#!/usr/bin/env python3
"""
Format a flat list of sources into structured markdown buckets.
Takes raw sources (books, articles, interviews, links) and organizes them
into a well-structured episode sources document.
"""
import argparse
from pathlib import Path
import os
from google import generativeai
import re
from dataclasses import dataclass
from typing import List, Dict
from urllib.parse import urlparse

@dataclass
class SourceItem:
    original_text: str
    source_type: str  # 'book', 'interview', 'link', 'article', 'person'
    category: str     # subcategory like 'historical', 'technical', etc.
    formatted_text: str
    author: str = ""
    title: str = ""
    date: str = ""
    url: str = ""

class SourcesFormatter:
    """AI-powered source list formatter and categorizer"""
    
    def __init__(self, api_key: str):
        generativeai.configure(api_key=api_key)
        self.model = generativeai.GenerativeModel("gemini-2.5-flash")
    
    def classify_sources(self, sources_text: str) -> List[SourceItem]:
        """Classify and format raw sources into structured categories"""
        
        prompt = f"""You are organizing sources for a podcast episode. Take this raw list of sources and classify each one into appropriate categories.

Raw sources:
{sources_text}

For each source, identify:
1. Source type: book, interview, link, article, person, other
2. Category: historical, technical, biographical, data, reference, etc.
3. Proper formatting with author/title/date if applicable
4. Extract any URLs if present

Return as JSON array:
[{{
  "original_text": "exact original text",
  "source_type": "book|interview|link|article|person|other",
  "category": "descriptive category",
  "formatted_text": "properly formatted citation",
  "author": "author name if applicable",
  "title": "title if applicable", 
  "date": "date if present",
  "url": "url if present"
}}]

Guidelines:
- Books: Format as "Author. Title"
- Interviews: Format as "Person Name, Date" 
- Links: Use descriptive title, keep URL
- Articles: "Author. Title (Publication)"
- People: Just the name
- Categorize by topic/theme (historical, technical, social, etc.)
"""
        
        try:
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            import json
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if json_match:
                sources_data = json.loads(json_match.group())
                
                source_items = []
                for source in sources_data:
                    source_items.append(SourceItem(
                        original_text=source.get('original_text', ''),
                        source_type=source.get('source_type', 'other'),
                        category=source.get('category', 'general'),
                        formatted_text=source.get('formatted_text', source.get('original_text', '')),
                        author=source.get('author', ''),
                        title=source.get('title', ''),
                        date=source.get('date', ''),
                        url=source.get('url', '')
                    ))
                
                return source_items
                
        except Exception as e:
            print(f"Error classifying sources: {e}")
            
            # Fallback: Basic classification
            lines = sources_text.strip().split('\n')
            source_items = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Basic heuristics
                if 'http' in line.lower():
                    source_type = 'link'
                    category = 'reference'
                elif any(word in line.lower() for word in ['interview', 'spoke with', 'conversation']):
                    source_type = 'interview'
                    category = 'interview'
                elif '.' in line and len(line.split('.')) >= 2:
                    if any(ext in line.lower() for ext in ['.com', '.org', '.edu']):
                        source_type = 'link'
                        category = 'reference'
                    else:
                        source_type = 'book'
                        category = 'historical'
                else:
                    source_type = 'other'
                    category = 'general'
                
                source_items.append(SourceItem(
                    original_text=line,
                    source_type=source_type,
                    category=category,
                    formatted_text=line
                ))
            
            return source_items
    
    def generate_markdown(self, source_items: List[SourceItem], episode_title: str = "Episode") -> str:
        """Generate structured markdown from classified sources"""
        
        # Group sources by type
        books = [s for s in source_items if s.source_type == 'book']
        interviews = [s for s in source_items if s.source_type == 'interview']
        links = [s for s in source_items if s.source_type == 'link']
        articles = [s for s in source_items if s.source_type == 'article']
        people = [s for s in source_items if s.source_type == 'person']
        other = [s for s in source_items if s.source_type not in ['book', 'interview', 'link', 'article', 'person']]
        
        markdown = f"# {episode_title} Sources\n\n"
        
        # Books section
        if books:
            markdown += "## Key Books\n\n"
            for book in books:
                markdown += f"{book.formatted_text}\n\n"
        
        # Interviews section
        if interviews:
            markdown += "## Interviews\n\n"
            for interview in interviews:
                markdown += f"{interview.formatted_text}\n\n"
        
        # Links section - organized by category
        if links:
            markdown += "## Key Links\n\n"
            
            # Group links by category
            link_categories = {}
            for link in links:
                category = link.category.title()
                if category not in link_categories:
                    link_categories[category] = []
                link_categories[category].append(link)
            
            for category, category_links in sorted(link_categories.items()):
                markdown += f"### {category}\n\n"
                for link in category_links:
                    if link.url:
                        markdown += f"[{link.title or link.formatted_text}]({link.url})\n\n"
                    else:
                        markdown += f"{link.formatted_text}\n\n"
        
        # Articles section
        if articles:
            markdown += "## Articles & Papers\n\n"
            for article in articles:
                if article.url:
                    markdown += f"[{article.formatted_text}]({article.url})\n\n"
                else:
                    markdown += f"{article.formatted_text}\n\n"
        
        # People section
        if people:
            markdown += "## Key People\n\n"
            for person in people:
                markdown += f"{person.formatted_text}\n\n"
        
        # Other sources
        if other:
            markdown += "## Other Sources\n\n"
            for item in other:
                markdown += f"{item.formatted_text}\n\n"
        
        return markdown

def main():
    parser = argparse.ArgumentParser(description="Format raw sources into structured markdown")
    parser.add_argument("sources_file", help="Path to file containing raw sources list")
    parser.add_argument("--output", "-o", help="Output markdown file path")
    parser.add_argument("--episode-title", default="Episode", help="Episode title for the sources document")
    parser.add_argument("--google-api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    sources_path = Path(args.sources_file)
    if not sources_path.exists():
        print(f"Sources file not found: {sources_path}")
        return
    
    # Get API key
    api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY found. Set it via --google-api-key or environment variable.")
        return
    
    # Read sources
    sources_text = sources_path.read_text()
    print(f"Processing sources from: {sources_path}")
    
    # Format sources
    formatter = SourcesFormatter(api_key)
    print("Classifying and formatting sources...")
    source_items = formatter.classify_sources(sources_text)
    
    print(f"Found {len(source_items)} sources:")
    type_counts = {}
    for item in source_items:
        type_counts[item.source_type] = type_counts.get(item.source_type, 0) + 1
    
    for source_type, count in sorted(type_counts.items()):
        print(f"  {count} {source_type}s")
    
    # Generate markdown
    print("\nGenerating structured markdown...")
    markdown = formatter.generate_markdown(source_items, args.episode_title)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = sources_path.with_suffix('.sources.md')
    
    # Write output
    output_path.write_text(markdown)
    print(f"\n📝 Formatted sources written to: {output_path}")
    
    # Show preview
    print(f"\n=== PREVIEW ===")
    preview_lines = markdown.split('\n')[:20]
    for line in preview_lines:
        print(line)
    if len(markdown.split('\n')) > 20:
        print(f"... ({len(markdown.split('\n')) - 20} more lines)")

if __name__ == "__main__":
    main()