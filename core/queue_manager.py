# core/queue_manager.py
# ─────────────────────────────────────────────
# Manages the URL queue (queue.json)
# Tracks: pending / processing / done / failed
# ─────────────────────────────────────────────

import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


# ── Queue Item Status ─────────────────────────
class Status:
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"
    SKIPPED    = "skipped"


@dataclass
class QueueItem:
    """One item in the queue"""
    url:          str
    topic:        str
    title:        str        = ""
    status:       str        = Status.PENDING
    added_at:     str        = ""
    processed_at: str        = ""
    error:        str        = ""
    video_id:     str        = ""
    chunks:       int        = 0
    words:        int        = 0
    retry_count:  int        = 0

    def to_dict(self) -> dict:
        return asdict(self)


class QueueManager:
    """
    Manages the URL processing queue

    Queue File: data/queue.json
    ─────────────────────────────────────────
    [
      {
        "url":    "https://youtube.com/...",
        "topic":  "AI agents tutorial",
        "status": "pending",
        ...
      }
    ]
    ─────────────────────────────────────────

    States:
    pending    → waiting to be processed
    processing → currently being ingested
    done       → successfully ingested
    failed     → error occurred
    skipped    → duplicate / no transcript
    """

    def __init__(
        self,
        queue_file: str = "data/queue.json"
    ):
        self.queue_file = queue_file
        Path("data").mkdir(exist_ok=True)
        self._ensure_file()

    # ── CORE OPERATIONS ───────────────────────

    def add(
        self,
        url:   str,
        topic: str,
        title: str = ""
    ) -> bool:
        """
        Add URL to queue
        Returns False if already exists
        """
        queue = self._load()

        # Check for duplicates
        existing_urls = {item["url"] for item in queue}
        if url in existing_urls:
            print(f"  ⏭️  Already in queue: {url[:50]}")
            return False

        # Create new item
        item = QueueItem(
            url      = url,
            topic    = topic,
            title    = title,
            status   = Status.PENDING,
            added_at = datetime.now().isoformat()
        )

        queue.append(item.to_dict())
        self._save(queue)
        print(f"  ✅ Added to queue: {title[:50] or url[:50]}")
        return True

    def add_batch(
        self,
        results: list,
        topic:   str
    ) -> int:
        """Add multiple search results to queue"""
        added = 0
        for result in results:
            success = self.add(
                url   = result.url,
                topic = topic,
                title = result.title
            )
            if success:
                added += 1
        return added

    def get_next_pending(self) -> Optional[dict]:
        """Get next pending item and mark as processing"""
        queue = self._load()

        for item in queue:
            if item["status"] == Status.PENDING:
                item["status"]    = Status.PROCESSING
                item["processed_at"] = datetime.now().isoformat()
                self._save(queue)
                return item

        return None     # No pending items

    def get_all_pending(self) -> list[dict]:
        """Get all pending items"""
        queue = self._load()
        return [
            item for item in queue
            if item["status"] == Status.PENDING
        ]

    def mark_done(
        self,
        url:      str,
        chunks:   int = 0,
        words:    int = 0,
        video_id: str = ""
    ):
        """Mark URL as successfully processed"""
        self._update(url, {
            "status":       Status.DONE,
            "processed_at": datetime.now().isoformat(),
            "chunks":       chunks,
            "words":        words,
            "video_id":     video_id
        })

    def mark_failed(self, url: str, error: str):
        """Mark URL as failed"""
        queue = self._load()
        for item in queue:
            if item["url"] == url:
                item["status"]       = Status.FAILED
                item["error"]        = str(error)[:200]
                item["retry_count"]  = item.get("retry_count", 0) + 1

                # Reset to pending if under retry limit
                if item["retry_count"] < 3:
                    item["status"] = Status.PENDING
                    print(f"  🔄 Will retry ({item['retry_count']}/3)")

                break
        self._save(queue)

    def mark_skipped(self, url: str, reason: str = ""):
        """Mark URL as skipped"""
        self._update(url, {
            "status": Status.SKIPPED,
            "error":  reason
        })

    def reset_processing(self):
        """
        Reset stuck 'processing' items back to pending
        Call this on startup to recover from crashes
        """
        queue   = self._load()
        reset_n = 0

        for item in queue:
            if item["status"] == Status.PROCESSING:
                item["status"] = Status.PENDING
                reset_n += 1

        if reset_n:
            self._save(queue)
            print(f"  🔄 Reset {reset_n} stuck items to pending")

    def reset_failed(self):
        """Reset all failed items back to pending (for retry)"""
        queue   = self._load()
        reset_n = 0

        for item in queue:
            if item["status"] == Status.FAILED:
                item["status"]      = Status.PENDING
                item["retry_count"] = 0
                item["error"]       = ""
                reset_n += 1

        self._save(queue)
        print(f"  🔄 Reset {reset_n} failed items to pending")

    # ── STATISTICS ────────────────────────────

    def get_stats(self) -> dict:
        """Get queue statistics"""
        queue = self._load()

        stats = {
            Status.PENDING:    0,
            Status.PROCESSING: 0,
            Status.DONE:       0,
            Status.FAILED:     0,
            Status.SKIPPED:    0,
            "total":           len(queue),
            "total_chunks":    0,
            "total_words":     0
        }

        for item in queue:
            status = item.get("status", Status.PENDING)
            if status in stats:
                stats[status] += 1
            stats["total_chunks"] += item.get("chunks", 0)
            stats["total_words"]  += item.get("words", 0)

        return stats

    def display_stats(self):
        """Print queue statistics"""
        s = self.get_stats()
        print(f"""
📊 Queue Statistics
{'─'*40}
Total     : {s['total']}
Pending   : {s[Status.PENDING]}
Done      : {s[Status.DONE]}
Failed    : {s[Status.FAILED]}
Skipped   : {s[Status.SKIPPED]}
{'─'*40}
Chunks    : {s['total_chunks']:,}
Words     : {s['total_words']:,}
{'─'*40}""")

    def display_queue(self, status_filter: str = None):
        """Display all queue items"""
        queue = self._load()

        if status_filter:
            queue = [
                q for q in queue
                if q["status"] == status_filter
            ]

        icons = {
            Status.PENDING:    "⏳",
            Status.PROCESSING: "🔄",
            Status.DONE:       "✅",
            Status.FAILED:     "❌",
            Status.SKIPPED:    "⏭️"
        }

        print(f"\n{'='*60}")
        print(f"📋 Queue ({len(queue)} items)")
        print(f"{'='*60}")

        for item in queue:
            icon  = icons.get(item["status"], "?")
            title = item.get("title", item["url"])[:45]
            print(
                f"{icon} [{item['status']:10}] "
                f"{title}"
            )

        print(f"{'='*60}")

    # ── FILE OPERATIONS ───────────────────────

    def _load(self) -> list:
        """Load queue from JSON file"""
        try:
            with open(self.queue_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self, queue: list):
        """Save queue to JSON file"""
        with open(self.queue_file, "w") as f:
            json.dump(queue, f, indent=2)

    def _ensure_file(self):
        """Create queue file if not exists"""
        if not os.path.exists(self.queue_file):
            self._save([])

    def _update(self, url: str, updates: dict):
        """Update specific item in queue"""
        queue = self._load()
        for item in queue:
            if item["url"] == url:
                item.update(updates)
                break
        self._save(queue)