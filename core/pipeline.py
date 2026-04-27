# core/pipeline.py
# ─────────────────────────────────────────────
# Connects all 4 layers into one command
# THIS IS YOUR RESEARCH AGENT!
# ─────────────────────────────────────────────

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.search        import YouTubeSearcher
from core.queue_manager import QueueManager
from core.worker        import IngestionWorker


class ResearchPipeline:

    def __init__(self):
        self.searcher = YouTubeSearcher()
        self.queue    = QueueManager()
        self.worker   = IngestionWorker()

        
    def run(
        self,
        topic:        str,
        max_videos:   int  = 5,
        auto_ingest:  bool = True,
        min_duration: int  = 180    # 3 minutes minimum
    ) -> dict:
        

        #PRINT HEADRER 
        print(f"""
╔══════════════════════════════════════════════╗
  🧠 AI Research Pipeline
  Topic: {topic}
╚══════════════════════════════════════════════╝
        """)

        # CREATE REPORT 
        report = { 
            "topic":        topic,
            "found":        0,
            "queued":       0,
            "ingested":     0,
            "failed":       0,
            "total_chunks": 0,
        }
         

         # RUN SEARCHING
        # ── STAGE 1: SEARCH ───────────────────
        print("📡 STAGE 1: Searching YouTube...")
        results = self.searcher.search( # call method search store in result 
            topic        = topic,    
            max_results  = max_videos,
            min_duration = min_duration
        )

        #ALGORITHM OF SEACHING (take user topic -> Search user topic -> return list of video result -> filter the video short than min_duration)
        


        # CHECK THE RESULT OF SEARCHING S
        if not results:
            print("❌ No results found! Try different topic.")
            return report
        #algorithm s
        # show found video to user 
        # store number of video found 
        self.searcher.display_results(results)
        report["found"] = len(results) # size of result data set 


        # ── STAGE 2: QUEUE ────────────────────
        print(f"\n📦 STAGE 2: Adding to Queue...")
        added = self.queue.add_batch(results, topic)
        report["queued"] = added
        print(f"✅ Added {added} new URLs to queue")

        # Show queue state
        self.queue.display_stats()

        # ── STAGE 3: INGEST ───────────────────
        if auto_ingest and added > 0:
            print(f"\n⚙️  STAGE 3: Auto-Ingesting...")
            worker_report     = self.worker.run_once()
            report["ingested"] = worker_report["processed"]
            report["failed"]   = worker_report["failed"]

        else:
            print(f"\n💡 Run worker manually:")
            print(f"   python core/worker.py")

        # ── FINAL REPORT ──────────────────────
        stats = self.queue.get_stats()
        report["total_chunks"] = stats["total_chunks"]

        self._print_report(report)
        return report

    def status(self):
        """Show current queue status"""
        self.queue.display_stats()
        self.queue.display_queue()

    def retry_failed(self):
        """Retry all failed items"""
        print("🔄 Retrying failed items...")
        self.queue.reset_failed()
        self.worker.run_once()

    def _print_report(self, report: dict):
        print(f"""
╔══════════════════════════════════════════════╗
  📊 Pipeline Complete!
╠══════════════════════════════════════════════╣
  Topic    : {report['topic']}
  Found    : {report['found']} videos
  Queued   : {report['queued']} new URLs
  Ingested : {report['ingested']} videos
  Failed   : {report['failed']} videos
  Chunks   : {report['total_chunks']:,} total
╚══════════════════════════════════════════════╝
        """)


# ══════════════════════════════════════════════
# ▶️ RUN IT DIRECTLY
# ══════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = ResearchPipeline()

    # Interactive mode
    print("🧠 AI Research Pipeline")
    print("=" * 40)

    topic = input("\nEnter research topic: ").strip()

    if not topic:
        print("❌ No topic entered!")
        sys.exit(1)

    max_v = input("Max videos? [default: 5]: ").strip()
    max_v = int(max_v) if max_v.isdigit() else 5 

    pipeline.run(
        topic      = topic,
        max_videos = max_v
    )