# utils.py
# ─────────────────────────────────────────────
# Helper functions used across the project
# ─────────────────────────────────────────────

import os
import re
import json
import hashlib
import tiktoken
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


# ══════════════════════════════════════════════
# 🎨 DISPLAY HELPERS
# ══════════════════════════════════════════════

def print_header(title: str):
    """Print beautiful section header"""
    console.print(Panel(
        f"[bold cyan]{title}[/bold cyan]",
        border_style="cyan"
    ))


def print_success(message: str):
    """Print success message"""
    console.print(f"[bold green]✅ {message}[/bold green]")


def print_error(message: str):
    """Print error message"""
    console.print(f"[bold red]❌ {message}[/bold red]")


def print_info(message: str):
    """Print info message"""
    console.print(f"[bold yellow]ℹ️  {message}[/bold yellow]")


def print_step(step: int, total: int, message: str):
    """Print step progress"""
    console.print(
        f"[cyan]Step {step}/{total}:[/cyan] "
        f"[white]{message}[/white]"
    )


# ══════════════════════════════════════════════
# 🔗 YOUTUBE URL HELPERS
# ══════════════════════════════════════════════

def extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from any URL format

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/shorts/VIDEO_ID
    """
    patterns = [
        r'(?:v=)([a-zA-Z0-9_-]{11})',      # Standard
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',  # Short
        r'(?:shorts/)([a-zA-Z0-9_-]{11})',   # Shorts
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(
        f"❌ Could not extract video ID from: {url}\n"
        f"Please use a valid YouTube URL"
    )


def create_video_id_hash(video_id: str) -> str:
    """Create short unique hash for storage"""
    return hashlib.md5(video_id.encode()).hexdigest()[:8]


# ══════════════════════════════════════════════
# 📝 TEXT PROCESSING HELPERS
# ══════════════════════════════════════════════

def clean_transcript(text: str) -> str:
    """
    Clean raw transcript text

    Removes:
    - Extra whitespace
    - Special characters
    - Repeated punctuation
    - Music/sound notation [Music] [Applause]
    """
    # Remove transcript artifacts
    text = re.sub(r'\[.*?\]', '', text)      # [Music] [Applause]
    text = re.sub(r'\(.*?\)', '', text)      # (inaudible)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)         # Multiple spaces
    text = re.sub(r'\n+', '\n', text)        # Multiple newlines

    # Clean punctuation artifacts
    text = re.sub(r'\.{3,}', '...', text)   # Multiple dots
    text = re.sub(r'-{2,}', '-', text)      # Multiple dashes

    return text.strip()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text (important for API costs)"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: estimate 4 chars per token
        return len(text) // 4

import os
import json

def save_chunks(video_id: str, chunks: list[dict]):
    os.makedirs("data/chunks", exist_ok=True)

    filepath = f"data/chunks/{video_id}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"💾 Chunks saved → {filepath}")

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Split text into overlapping chunks

    Why overlap?
    → Preserve context at chunk boundaries
    → Don't lose information at splits

    Returns list of chunks with metadata:
    [
        {
            "text": "chunk content...",
            "chunk_index": 0,
            "token_count": 120,
            "char_count": 480
        }
    ]
    """
    words = text.split()
    chunks = []
    chunk_index = 0

    i = 0
    while i < len(words):
        # Get chunk words
        chunk_words = words[i:i + chunk_size]
        chunk_text_content = " ".join(chunk_words)

        # Store chunk with metadata
        chunks.append({
            "text": chunk_text_content,
            "chunk_index": chunk_index,
            "token_count": count_tokens(chunk_text_content),
            "char_count": len(chunk_text_content),
            "word_count": len(chunk_words)
        })

        chunk_index += 1

        # Move forward (with overlap)
        i += chunk_size - overlap

    return chunks


# ══════════════════════════════════════════════
# 💾 FILE HELPERS
# ══════════════════════════════════════════════

def save_json(data: dict, filepath: str):
    """Save data as JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print_success(f"Saved: {filepath}")


def load_json(filepath: str) -> dict:
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_transcript(
    video_id: str,
    transcript_data: dict,
    base_dir: str = "data/transcripts"
):
    """Save transcript to file"""
    filepath = f"{base_dir}/{video_id}.json"
    save_json(transcript_data, filepath)
    return filepath


def load_transcript(
    video_id: str,
    base_dir: str = "data/transcripts"
) -> dict:
    """Load transcript from file"""
    filepath = f"{base_dir}/{video_id}.json"
    return load_json(filepath)


def transcript_exists(
    video_id: str,
    base_dir: str = "data/transcripts"
) -> bool:
    """Check if transcript already downloaded"""
    filepath = f"{base_dir}/{video_id}.json"
    return os.path.exists(filepath)


# ══════════════════════════════════════════════
# 📊 METADATA HELPERS
# ══════════════════════════════════════════════

def create_metadata(
    video_id: str,
    url: str,
    title: str = "Unknown",
    chunk_index: int = 0,
    total_chunks: int = 0
) -> dict:
    """Create metadata for vector store"""
    return {
        "video_id": video_id,
        "url": url,
        "title": title,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "ingested_at": datetime.now().isoformat(),
        "source": "youtube"
    }


def format_answer(
    answer: str,
    sources: list[dict],
    question: str
) -> str:
    """Format the final answer with sources"""
    source_text = "\n".join([
        f"  📍 Chunk {s.get('chunk_index', '?')} "
        f"from video: {s.get('video_id', 'Unknown')}"
        for s in sources[:3]
    ])

    return f"""
╔══════════════════════════════════════════════╗
  ❓ Question: {question}
╠══════════════════════════════════════════════╣
  🤖 Answer:
  {answer}
╠══════════════════════════════════════════════╣
  📚 Sources:
{source_text}
╚══════════════════════════════════════════════╝
    """.strip()