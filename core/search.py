# core/search.py
# ─────────────────────────────────────────────
# Searches YouTube and returns URLs
# Uses YouTube Data API v3 (free quota)
# Falls back to yt-dlp if no API key
# ─────────────────────────────────────────────

import os
import json
import requests
from dataclasses import dataclass


@dataclass
class SearchResult:
    """One search result"""
    url:       str
    title:     str
    channel:   str
    duration:  str
    views:     int
    topic:     str


class YouTubeSearcher:
    """
    Searches YouTube for videos on a topic

    Strategy:
    ──────────────────────────────────────────
    1. Try YouTube Data API v3 (best results)
    2. Fallback → yt-dlp  (no API key needed)
    3. Fallback → scraping (last resort)
    ──────────────────────────────────────────
    """

    def __init__(self):
        self.api_key  = os.getenv("YOUTUBE_API_KEY", "")
        self.base_url = "https://www.googleapis.com/youtube/v3"

    def search(
        self,
        topic:       str,
        max_results: int = 10,
        min_duration: int = 120    # seconds (2 min minimum)
    ) -> list[SearchResult]:
        """
        Main search method
        Returns list of SearchResult objects
        """
        print(f"\n🔍 Searching YouTube: '{topic}'")

        # Try API first
        if self.api_key:
            results = self._search_via_api(topic, max_results)
        else:
            # Fallback to yt-dlp
            results = self._search_via_ytdlp(topic, max_results)

        # Filter short videos
        filtered = [
            r for r in results
            if self._parse_duration(r.duration) >= min_duration
        ]

        print(f"✅ Found {len(filtered)} videos (filtered from {len(results)})")
        return filtered

    # ── METHOD 1: YouTube Data API ────────────
    def _search_via_api(
        self,
        topic:       str,
        max_results: int
    ) -> list[SearchResult]:
        """Search using official YouTube Data API v3"""

        print("  Using YouTube Data API v3...")

        # Step 1: Search for videos
        search_response = requests.get(
            f"{self.base_url}/search",
            params={
                "key":        self.api_key,
                "q":          topic,
                "part":       "snippet",
                "type":       "video",
                "maxResults": max_results,
                "relevanceLanguage": "en",
                "order":      "relevance"
            },
            timeout=10
        )
        search_response.raise_for_status()
        search_data = search_response.json()

        # Extract video IDs
        video_ids = [
            item["id"]["videoId"]
            for item in search_data.get("items", [])
        ]

        if not video_ids:
            return []

        # Step 2: Get video details (duration, views)
        details_response = requests.get(
            f"{self.base_url}/videos",
            params={
                "key":  self.api_key,
                "id":   ",".join(video_ids),
                "part": "snippet,contentDetails,statistics"
            },
            timeout=10
        )
        details_data = details_response.json()

        # Build results
        results = []
        for item in details_data.get("items", []):
            snippet  = item.get("snippet", {})
            details  = item.get("contentDetails", {})
            stats    = item.get("statistics", {})
            video_id = item["id"]

            results.append(SearchResult(
                url      = f"https://youtube.com/watch?v={video_id}",
                title    = snippet.get("title", ""),
                channel  = snippet.get("channelTitle", ""),
                duration = details.get("duration", "PT0S"),
                views    = int(stats.get("viewCount", 0)),
                topic    = topic
            ))

        return results

    # ── METHOD 2: yt-dlp fallback ─────────────
    def _search_via_ytdlp(
        self,
        topic:       str,
        max_results: int
    ) -> list[SearchResult]:
        """Search using yt-dlp (no API key needed)"""
        try:
            import yt_dlp

            print("  Using yt-dlp (no API key)...")

            ydl_opts = {
                "quiet":          True,
                "no_warnings":    True,
                "extract_flat":   True,
                "playlist_items": f"1:{max_results}"
            }

            search_url = (
                f"ytsearch{max_results}:{topic}"
            )

            results = []
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    search_url,
                    download=False
                )

                entries = info.get("entries", [])
                for entry in entries:
                    if not entry:
                        continue

                    video_id = entry.get("id", "")
                    duration = entry.get("duration", 0)

                    results.append(SearchResult(
                        url      = f"https://youtube.com/watch?v={video_id}",
                        title    = entry.get("title", ""),
                        channel  = entry.get("uploader", ""),
                        duration = self._seconds_to_iso(duration),
                        views    = entry.get("view_count", 0),
                        topic    = topic
                    ))

            return results

        except ImportError:
            print("  yt-dlp not installed!")
            print("  Run: pip install yt-dlp")
            return []
        except Exception as e:
            print(f"  yt-dlp search failed: {e}")
            return []

    # ── HELPERS ───────────────────────────────
    def _parse_duration(self, duration: str) -> int:
        """
        Convert ISO 8601 duration to seconds
        PT1H2M3S → 3723 seconds
        """
        import re

        if not duration or duration == "PT0S":
            return 0

        # Handle numeric (from yt-dlp)
        if isinstance(duration, (int, float)):
            return int(duration)

        pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
        match   = re.match(pattern, str(duration))

        if not match:
            return 0

        hours   = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    def _seconds_to_iso(self, seconds: int) -> str:
        """Convert seconds back to ISO 8601"""
        if not seconds:
            return "PT0S"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        result = "PT"
        if h: result += f"{h}H"
        if m: result += f"{m}M"
        if s: result += f"{s}S"
        return result

    def display_results(
        self,
        results: list[SearchResult]
    ):
        """Pretty print search results"""
        print(f"\n{'='*60}")
        print(f"📺 Search Results ({len(results)} videos)")
        print(f"{'='*60}")

        for i, r in enumerate(results, 1):
            secs    = self._parse_duration(r.duration)
            mins    = secs // 60
            print(f"""
[{i}] {r.title[:55]}
     Channel  : {r.channel}
     Duration : {mins} minutes
     Views    : {r.views:,}
     URL      : {r.url}""")

        print(f"{'='*60}")