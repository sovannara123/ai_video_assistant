# core/worker.py
# ─────────────────────────────────────────────
# Auto Ingestion Worker
# Reads queue → runs ingest → stores in ChromaDB
# ─────────────────────────────────────────────

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.queue_manager import QueueManager, Status
from ingest             import VideoIngester
from vector_store       import KnowledgeBase


class IngestionWorker:
    """
    Reads from queue and auto-ingests videos

    Modes:
    ────────────────────────────────────────
    run_once()  → Process all pending items
    run_loop()  → Keep running, check every N secs
    run_one()   → Process exactly one item
    ────────────────────────────────────────
    """

    def __init__(self):
        self.queue    = QueueManager()
        self.ingester = VideoIngester()
        self.kb       = KnowledgeBase()

        # Recover from previous crashes
        self.queue.reset_processing()

    def run_once(self) -> dict:
        """
        Process ALL pending items in queue
        Returns summary report
        """
        pending = self.queue.get_all_pending()

        if not pending:
            print("✅ Queue is empty! Nothing to process.")
            return {"processed": 0, "failed": 0}

        print(f"\n🚀 Processing {len(pending)} pending items...")
        print(f"{'='*50}")

        success_count = 0
        fail_count    = 0

        for i, item in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] Processing:")
            print(f"  Title : {item.get('title', 'Unknown')[:50]}")
            print(f"  URL   : {item['url'][:50]}")
            print(f"  Topic : {item['topic']}")

            result = self._process_item(item)

            if result["success"]:
                success_count += 1
                print(
                    f"  ✅ Done! "
                    f"Chunks: {result['chunks']}, "
                    f"Words: {result['words']}"
                )
            else:
                fail_count += 1
                print(f"  ❌ Failed: {result['error']}")

            # Small delay between videos
            if i < len(pending):
                time.sleep(2)

        # Final summary
        print(f"\n{'='*50}")
        print(f"📊 Worker Complete!")
        print(f"   ✅ Success : {success_count}")
        print(f"   ❌ Failed  : {fail_count}")
        print(f"{'='*50}")

        return {
            "processed": success_count,
            "failed":    fail_count
        }

    def run_loop(self, interval: int = 60):
        """
        Keep running and check for new items
        interval = seconds between checks
        """
        print(f"\n🔄 Worker running (checks every {interval}s)")
        print(f"   Press Ctrl+C to stop\n")

        try:
            while True:
                stats   = self.queue.get_stats()
                pending = stats[Status.PENDING]

                if pending > 0:
                    print(
                        f"\n⏰ Found {pending} pending items!"
                    )
                    self.run_once()
                else:
                    print(
                        f"⏳ Queue empty. "
                        f"Checking again in {interval}s...",
                        end="\r"
                    )

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n👋 Worker stopped!")

    def run_one(self) -> dict:
        """Process exactly one pending item"""
        item = self.queue.get_next_pending()

        if not item:
            print("✅ No pending items in queue!")
            return {"success": False, "error": "Queue empty"}

        print(f"\n🔄 Processing one item:")
        print(f"  {item.get('title', item['url'])[:60]}")

        return self._process_item(item)

    # ── CORE PROCESSING ───────────────────────

    def _process_item(self, item: dict) -> dict:
        """
        Process a single queue item

        Steps:
        1. Ingest (transcript → chunks)
        2. Embed (chunks → vectors)
        3. Store (vectors → ChromaDB)
        4. Update queue status
        """
        url   = item["url"]
        topic = item.get("topic", "general")

        try:
            # ── Step 1: Ingest ─────────────────
            ingestion = self.ingester.ingest(url)

            if ingestion["status"] != "success":
                raise Exception(
                    ingestion.get("error", "Ingestion failed")
                )

            # ── Step 2 & 3: Embed + Store ──────
            self.kb.add_video(ingestion)

            # ── Step 4: Mark done ──────────────
            self.queue.mark_done(
                url      = url,
                chunks   = ingestion["total_chunks"],
                words    = ingestion["word_count"],
                video_id = ingestion["video_id"]
            )

            return {
                "success":  True,
                "chunks":   ingestion["total_chunks"],
                "words":    ingestion["word_count"],
                "video_id": ingestion["video_id"]
            }

        except Exception as e:
            self.queue.mark_failed(url, str(e))
            return {
                "success": False,
                "error":   str(e)
            }