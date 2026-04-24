# ingest.py
# ─────────────────────────────────────────────
# 🛡️ PATCHED: Handles None, empty transcripts,
# and malformed API responses gracefully
# ─────────────────────────────────────────────

from config import config
from utils import (
    extract_video_id,
    clean_transcript,
    chunk_text,
    save_transcript,
    load_transcript,
    transcript_exists,
    print_header,
    print_success,
    print_error,
    print_info,
    print_step,
    save_chunks,
    console
)


# this function does take a youtube video , 
#Returns a list of transcript segments (text + time)
def fetch_transcript_v1(video_id: str) -> list[dict]: # param video_id type as string and return list of dict
    """Fetch transcript with strict None handling"""
    from youtube_transcript_api import YouTubeTranscriptApi 


     
    print_info("Using youtube-transcript-api v1.2.4")
    api = YouTubeTranscriptApi() # create api object to talk to youtube subtitle 

   

    # try to get english subtitles first, but handle cases where they don't exist
    try:
        fetched = api.fetch(video_id, languages=['en', 'en-US', 'en-GB']) # here fetch in english first if not exist throw the error and move to the next method
        result  = _parse_fetched(fetched)  # safely parse the fetched result into a list of dicts with text, start, and duration
        if result and len(result) > 0: #
            print_success(f"✅ Fetched with languages! ({len(result)} segments)")
            return result # this result type as list of dicts with text, start, and duration
    except Exception as e:
        print_info(f"Method 1 failed: {e}")
    

    # ── Method 2: Fetch without filter ────────────
    try:
        fetched = api.fetch(video_id) # no luage filter, just fetch whatever is available
        result  = _parse_fetched(fetched)
        if result and len(result) > 0:
            print_success(f"✅ Fetched without filter! ({len(result)} segments)")
            return result
    except Exception as e:
        print_info(f"Method 2 failed: {e}")

    # ── Method 3: List then fetch ──────────────────
    try:
        transcript_list = api.list(video_id) # list all available transcripts for the video, this is useful to handle cases where the API response structure has changed in newer versions
        print_info(f"Available: {[t.language_code for t in transcript_list]}")

        #Try each one until one works
        for transcript in transcript_list:
            try:
                fetched = transcript.fetch()
                result  = _parse_fetched(fetched)
                if result and len(result) > 0:
                    print_success(f"✅ Fetched via list! Lang: {transcript.language_code}")
                    return result
            except Exception:
                continue
    except Exception as e:
        print_info(f"Method 3 failed: {e}")


    #If everything fails
    # stop the program with massage 
    raise ValueError(
        f"❌ No transcript available for video: {video_id}\n"
        f"Possible reasons:\n"
        f"  → No English subtitles exist\n"
        f"  → Auto-generated captions not ready\n"
        f"  → Video is private/age-restricted/live\n"
        f"  → Try a different YouTube video"
    )

# ══════════════════════════════════════════════
# 🛠️ HELPER: Safe parsing of fetched transcript
def _parse_fetched(fetched) -> list[dict]:
    """
    Safely parse FetchedTranscript → list of dicts
    Handles None, empty, and malformed responses
    """
    if fetched is None: # python concept None checking , check if the fetched result is none 
        return []

    result = []   # create list to store cleaned transcript

    try:
        # Direct list fallback
        if isinstance(fetched, list): 
            for item in fetched:
                if isinstance(item, dict):
                    result.append({
                        "text":     str(item.get('text', '')).strip(),
                        "start":    float(item.get('start', 0)),
                        "duration": float(item.get('duration', 0))
                    })
        # Iterable object (v1.2.x FetchedTranscript)
        elif hasattr(fetched, '__iter__'): # check if the fetched object is iterable, this is important because the API response structure may have changed in newer versions, and we want to be able to handle both the old and new structures gracefully
            for item in fetched: #
                if hasattr(item, 'text'):# check if the item has a text attribute, this is important because the API response structure may have changed in newer versions, and we want to be able to handle both the old and new structures gracefully
                    result.append({ 
                        "text":     str(item.text).strip(), #Convert data into correct format. , Strip () : remove space from text 
                        "start":    float(getattr(item, 'start', 0)), # if start attribute does not exist , use default value 0
                        "duration": float(getattr(item, 'duration', 0))
                    })
                elif isinstance(item, dict): #check type of vairiable item is dict ,
                    result.append({
                        "text":     str(item.get('text', '')).strip(),
                        "start":    float(item.get('start', 0)),
                        "duration": float(item.get('duration', 0))
                    })

    except Exception as e:
        print_info(f"⚠️ Parse warning: {e}")

    # Filter out empty segments OR empty subtitle like ""
    return [seg for seg in result if seg.get("text")] # List comprehension 

    
# ══════════════════════════════════════════════
# 📥 TRANSCRIPT FETCHER (PATCHED)
# ══════════════════════════════════════════════
# a machine that fetches transcripts”
class TranscriptFetcher:
    def fetch(self, url: str) -> dict:
        print_header("📥 Fetching YouTube Transcript")

        print_step(1, 4, "Extracting video ID...")
        video_id = extract_video_id(url)
        print_success(f"Video ID: {video_id}")

        print_step(2, 4, "Checking cache...") # have you alraedy download the transcript
        if transcript_exists(video_id):
            print_info("Found in cache! Loading...")
            return load_transcript(video_id)
        print_info("Not cached, downloading...")

        print_step(3, 4, "Downloading transcript...")
        raw_transcript = fetch_transcript_v1(video_id)

        # 🛡️ CRITICAL FIX: Guard against None/empty. meaning “If data is empty OR wrong type → stop”
        if not raw_transcript or not isinstance(raw_transcript, list):
            raise ValueError(
                f"Transcript fetch returned invalid data: {type(raw_transcript)}"
            ) 

        print_step(4, 4, "Processing transcript...")
        processed = self._process(video_id, url, raw_transcript) # method call another method inside class

        save_transcript(video_id, processed)

        print_success(
            f"Cached! {processed['segment_count']} segments, "
            f"{processed['duration_seconds']:.0f}s duration"
        )
        return processed

    def _process(self, video_id: str, url: str, raw: list) -> dict:
        # 🛡️ CRITICAL FIX: Validate raw before len()
        if not raw:
            raise ValueError("Raw transcript is empty!")

        full_text = " ".join([
            seg["text"] for seg in raw 
            if seg.get("text", "").strip()
        ]).strip() 

        if not full_text:
            raise ValueError("Transcript contains no usable text!")

        full_text = clean_transcript(full_text)

        duration = 0
        if raw:
            last = raw[-1]
            duration = last.get("start", 0) + last.get("duration", 0)

        return {
            "video_id":         video_id,
            "url":              url,
            "raw_transcript":   raw,
            "full_text":        full_text,
            "duration_seconds": duration,
            "segment_count":    len(raw),
            "word_count":       len(full_text.split()),
            "char_count":       len(full_text)
        }


# ══════════════════════════════════════════════
# ✂️ CHUNKER & PIPELINE (UNCHANGED BUT SAFE)
# ══════════════════════════════════════════════

class TranscriptChunker: # 
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap    = overlap    or config.CHUNK_OVERLAP 

    def chunk(self, transcript_data: dict) -> list[dict]:
        print_header("✂️  Chunking Transcript")
        video_id  = transcript_data["video_id"]
        full_text = transcript_data["full_text"]

        print_info(
            f"Words: {transcript_data['word_count']:,} | "
            f"Chunk: {self.chunk_size} | Overlap: {self.overlap}"
        )

        raw_chunks = chunk_text(full_text, self.chunk_size, self.overlap)

        enriched = []
        for chunk in raw_chunks:
            enriched.append({
                "id": f"{video_id}_chunk_{chunk['chunk_index']}",
                "text": chunk["text"],
                "metadata": {
                    "video_id": video_id,
                    "url": transcript_data["url"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": len(raw_chunks),
                    "token_count": chunk["token_count"],
                    "word_count": chunk["word_count"],
                    "video_duration": transcript_data["duration_seconds"]
                }
            })

        print_success(f"Created {len(enriched)} chunks!")
        return enriched


class VideoIngester:
    def __init__(self):
        self.fetcher = TranscriptFetcher()
        self.chunker = TranscriptChunker()

    def ingest(self, url: str) -> dict:
        console.rule("[bold cyan]🚀 Starting Ingestion[/bold cyan]")
        try:
            transcript_data = self.fetcher.fetch(url)

            # 🛡️ Validate before chunking
            if not transcript_data.get("full_text", "").strip():
                raise ValueError("Transcript is empty after processing!")
            if transcript_data.get("word_count", 0) < 10:
                raise ValueError(f"Transcript too short ({transcript_data['word_count']} words)!")

            chunks = self.chunker.chunk(transcript_data)

            # 💾 NEW: save chunks to disk

            save_chunks(transcript_data["video_id"], chunks)

            summary = {
                "video_id": transcript_data["video_id"],
                "url": url,
                "total_chunks": len(chunks),
                "word_count": transcript_data["word_count"],
                "duration_seconds": transcript_data["duration_seconds"],
                "chunks": chunks,
                "status": "success"
            }

            console.rule("[bold green]✅ Done![/bold green]")
            self._print_summary(summary)
            return summary

        except Exception as e:
            print_error(f"Ingestion failed: {e}")
            return {"status": "failed", "error": str(e), "url": url}

    def _print_summary(self, s: dict):
        mins = s['duration_seconds'] / 60
        console.print(f"""
[bold]📊 Ingestion Summary[/bold]
──────────────────────────
Video ID : [cyan]{s['video_id']}[/cyan]
Duration : {mins:.1f} minutes
Words    : {s['word_count']:,}
Chunks   : {s['total_chunks']}
Status   : [green]✅ {s['status']}[/green]
──────────────────────────
        """)


if __name__ == "__main__":
    print("🧪 Testing patched ingest.py\n")
    url = input("Paste YouTube URL: ").strip()
    ingester = VideoIngester()
    result = ingester.ingest(url)
    if result["status"] == "success":
        print_success("Pipeline working!")
        print(f"\nFirst chunk preview:\n{result['chunks'][0]['text'][:300]}")
    else:
        print_error(f"Failed: {result['error']}")