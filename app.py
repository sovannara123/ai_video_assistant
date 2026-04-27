# app.py
# ─────────────────────────────────────────────
# Main Application — Full Version
# ─────────────────────────────────────────────

import sys
import os
import ollama

from config       import config
from ingest       import VideoIngester
from vector_store import KnowledgeBase
from utils        import (
    print_header,
    print_success,
    print_error,
    print_info,
    console
)
from rich.prompt  import Prompt
from rich.panel   import Panel
from rich.table   import Table


# ══════════════════════════════════════════════
# 🤖 RAG ENGINE
# ══════════════════════════════════════════════

class RAGEngine:
    """
    RAG Engine using Ollama (Local LLM)

    Flow:
    Question → Embed → Search → Context → Ollama → Answer
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb           = knowledge_base
        self.model        = config.OLLAMA_MODEL
        self.chat_history = []

    def ask(
        self,
        question:  str,
        video_id:  str = None,
        n_context: int = 5
    ) -> dict:
        """Main RAG pipeline"""

        print_info(f"🔍 Searching knowledge base...")

        # Step 1: Retrieve relevant chunks
        relevant_chunks = self.kb.search(
            query     = question,
            n_results = n_context,
            video_id  = video_id
        )

        if not relevant_chunks:
            return {
                "question":     question,
                "answer": (
                    "I couldn't find relevant information.\n"
                    "Please add videos first using 'add' or "
                    "'research' command."
                ),
                "sources":      [],
                "context_used": 0,
                "model_used":   self.model
            }

        # Step 2: Build context
        context = self._build_context(relevant_chunks)

        # Step 3: Generate with Ollama
        print_info(f"🦙 Generating answer with {self.model}...")
        answer = self._generate_with_ollama(question, context)

        # Step 4: Save history
        self.chat_history.append({
            "question": question,
            "answer":   answer,
            "model":    self.model
        })

        return {
            "question":     question,
            "answer":       answer,
            "sources":      [c["metadata"] for c in relevant_chunks],
            "context_used": len(relevant_chunks),
            "model_used":   self.model
        }

    def _build_context(self, chunks: list[dict]) -> str:
        """Build context string from chunks"""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            sim = chunk.get("similarity", 0)
            parts.append(
                f"[Excerpt {i} | Relevance: {sim:.0%}]\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # app.py
# ─────────────────────────────────────────────
# FIND THIS METHOD — Replace options dict
# ─────────────────────────────────────────────

    def _generate_with_ollama(
        self,
        question: str,
        context:  str
    ) -> str:
        """Generate answer using human-like AI settings"""

        prompt = f"""
    {config.SYSTEM_PROMPT}

    Context from video transcripts:
    {context}

    Question: {question}

    Answer:"""

        try:
            response = ollama.generate(
                model   = self.model,
                prompt  = prompt,
                options = {
                    "temperature":       config.TEMPERATURE,
                    "top_p":             config.TOP_P,
                    "repetition_penalty": config.REPETITION_PENALTY,
                    "frequency_penalty":  config.FREQUENCY_PENALTY,
                    "presence_penalty":   config.PRESENCE_PENALTY,
                    "num_predict":        config.MAX_OUTPUT_TOKENS,
                }
            )
            return response["response"]

        except Exception as e:
            print_error(f"Ollama error: {e}")
            return (
                f"Error generating answer: {e}\n"
                f"Make sure Ollama is running: ollama serve"
            )


    # ══════════════════════════════════════════════
    # ALSO UPDATE STREAM_ASK METHOD
    # ══════════════════════════════════════════════

    def stream_ask(
        self,
        question: str,
        video_id: str = None
    ):
        """Stream answer token by token (with human-like settings)"""
        relevant_chunks = self.kb.search(
            query     = question,
            n_results = 5,
            video_id  = video_id
        )

        if not relevant_chunks:
            print_error("No relevant content found!")
            return

        context = self._build_context(relevant_chunks)

        prompt = f"""
    {config.SYSTEM_PROMPT}

    Context:
    {context}

    Question: {question}

    Answer:"""

        print_info(f"🦙 {self.model} is thinking...\n")
        console.print("[bold green]🤖 Answer:[/bold green]")

        for chunk in ollama.generate(
            model   = self.model,
            prompt  = prompt,
            stream  = True,
            options = {
                "temperature":       config.TEMPERATURE,
                "top_p":             config.TOP_P,
                "repetition_penalty": config.REPETITION_PENALTY,
                "frequency_penalty":  config.FREQUENCY_PENALTY,
                "presence_penalty":   config.PRESENCE_PENALTY,
            }
        ):
            print(chunk["response"], end="", flush=True)

        print("\n")

    def summarize_video(self, video_id: str) -> str:
        """Summarize entire video"""
        return self.ask(
            question = (
                "Give me a comprehensive summary. "
                "What are the main topics, key points, "
                "and important takeaways from this video?"
            ),
            video_id  = video_id,
            n_context = 10
        )["answer"]

    def extract_workflow(self, video_id: str) -> str:
        """Extract workflows and tutorials"""
        return self.ask(
            question = (
                "Extract step-by-step workflows, processes, "
                "or tutorials. Format as numbered steps."
            ),
            video_id  = video_id,
            n_context = 8
        )["answer"]

    def list_available_models(self) -> list[str]:
        """List all Ollama models available"""
        try:
            return config._get_ollama_models()
        except Exception:
            return []

    def switch_model(self, model_name: str) -> bool:
        """Switch to different Ollama model"""
        available = self.list_available_models()

        if model_name not in available:
            print_error(
                f"Model '{model_name}' not found!\n"
                f"Available: {available}\n"
                f"Install: ollama pull {model_name}"
            )
            return False

        self.model = model_name
        print_success(f"Switched to model: {model_name}")
        return True


# ══════════════════════════════════════════════
# 🖥️ MAIN APPLICATION
# ══════════════════════════════════════════════

class AIVideoAssistant:
    """
    Main AI Video Learning Assistant
    100% Open Source & Free!
    """

    def __init__(self):
        config.validate()
        self.ingester = VideoIngester()
        self.kb       = KnowledgeBase()
        self.rag      = RAGEngine(self.kb)

    def run(self):
        """Start interactive app"""
        self._print_welcome()

        # All available commands
        commands = [
            "research",
            "add",
            "ask",
            "stream",
            "list",
            "queue",
            "retry",
            "summary",
            "workflow",
            "models",
            "switch",
            "stats",
            "quit"
        ]

        # Map commands to handler methods
        handlers = {
            "research": self._handle_research,
            "add":      self._handle_add,
            "ask":      self._handle_ask,
            "stream":   self._handle_stream,
            "list":     self._handle_list,
            "queue":    self._handle_queue,
            "retry":    self._handle_retry,
            "summary":  self._handle_summary,
            "workflow": self._handle_workflow,
            "models":   self._handle_models,
            "switch":   self._handle_switch,
            "stats":    self._handle_stats,
            "quit":     self._handle_quit,
        }

        while True:
            try:
                command = Prompt.ask(
                    "\n[bold cyan]Command[/bold cyan]",
                    choices = commands,
                    default = "ask"
                )

                handler = handlers.get(command)
                if handler:
                    result = handler()
                    if result == "quit":
                        break

            except KeyboardInterrupt:
                console.print(
                    "\n[bold yellow]👋 Goodbye![/bold yellow]"
                )
                break
            except Exception as e:
                print_error(f"Error: {e}")

    # ══════════════════════════════════════════
    # 🔬 COMMAND HANDLERS
    # ══════════════════════════════════════════

    def _handle_research(self):
       
        try:
            from core.pipeline import ResearchPipeline

            topic = Prompt.ask(
                "[cyan]Research topic[/cyan]"
            )

            if not topic.strip():
                print_error("Topic cannot be empty!")
                return

            max_v = Prompt.ask(
                "[cyan]Max videos to collect[/cyan]",
                default = "5"
            )

            max_v = int(max_v) if max_v.isdigit() else 5

            pipeline = ResearchPipeline()
            pipeline.run(
                topic      = topic.strip(),
                max_videos = max_v
            )

        except ImportError:
            print_error(
                "Pipeline not found!\n"
                "Make sure core/pipeline.py exists"
            )
        except Exception as e:
            print_error(f"Research failed: {e}")

    def _handle_add(self):
        """Add single YouTube video manually"""
        url = Prompt.ask("[cyan]YouTube URL[/cyan]")

        if not url.strip():
            print_error("URL cannot be empty!")
            return

        result = self.ingester.ingest(url.strip())

        if result["status"] == "success":
            self.kb.add_video(result)
        else:
            print_error(
                f"Failed: {result.get('error', 'Unknown error')}"
            )

    def _handle_ask(self):
        """Ask question — full response"""
        question = Prompt.ask("[cyan]Your question[/cyan]")

        if not question.strip():
            print_error("Question cannot be empty!")
            return

        result = self.rag.ask(question.strip())

        console.print(Panel(
            f"[white]{result['answer']}[/white]",
            title = (
                f"[bold green]"
                f"🤖 {result['model_used']}"
                f"[/bold green]"
            ),
            border_style = "green"
        ))

        print_info(
            f"Answer based on "
            f"{result['context_used']} video excerpts"
        )

    def _handle_stream(self):
        """Ask question with streaming response"""
        question = Prompt.ask("[cyan]Your question[/cyan]")

        if not question.strip():
            print_error("Question cannot be empty!")
            return

        self.rag.stream_ask(question.strip())

    def _handle_list(self):
        """List all indexed videos"""
        videos = self.kb.store.list_videos()

        if not videos:
            print_info(
                "No videos indexed yet!\n"
                "Use 'add' or 'research' to add videos."
            )
            return

        table = Table(title="📹 Indexed Videos")
        table.add_column("#",        style="dim",   width=4)
        table.add_column("Video ID", style="cyan")
        table.add_column("Status",   style="green")

        for i, video in enumerate(videos, 1):
            table.add_row(str(i), video, "✅ Indexed")

        console.print(table)
        print_info(f"Total: {len(videos)} video(s)")

    def _handle_queue(self):
        """Show queue status"""
        try:
            from core.queue_manager import QueueManager
            queue = QueueManager()
            queue.display_stats()
            queue.display_queue()
        except ImportError:
            print_error(
                "Queue manager not found!\n"
                "Make sure core/queue_manager.py exists"
            )
        except Exception as e:
            print_error(f"Queue error: {e}")

    def _handle_retry(self):
        """Retry all failed queue items"""
        try:
            from core.pipeline import ResearchPipeline
            print_info("Retrying all failed items...")
            pipeline = ResearchPipeline()
            pipeline.retry_failed()
        except ImportError:
            print_error(
                "Pipeline not found!\n"
                "Make sure core/pipeline.py exists"
            )
        except Exception as e:
            print_error(f"Retry failed: {e}")

    def _handle_summary(self):
        """Summarize a specific video"""
        videos = self.kb.store.list_videos()

        if not videos:
            print_info(
                "No videos indexed yet!\n"
                "Use 'add' or 'research' first."
            )
            return

        video_id = Prompt.ask(
            "[cyan]Video ID to summarize[/cyan]",
            choices = videos
        )

        print_info("Generating summary...")

        summary = self.rag.summarize_video(video_id)

        console.print(Panel(
            f"[white]{summary}[/white]",
            title = (
                f"[bold green]"
                f"📋 Summary: {video_id}"
                f"[/bold green]"
            ),
            border_style = "green"
        ))

    def _handle_workflow(self):
        """Extract workflow from a video"""
        videos = self.kb.store.list_videos()

        if not videos:
            print_info("No videos indexed yet!")
            return

        video_id = Prompt.ask(
            "[cyan]Video ID[/cyan]",
            choices = videos
        )

        print_info("Extracting workflow...")

        workflow = self.rag.extract_workflow(video_id)

        console.print(Panel(
            f"[white]{workflow}[/white]",
            title = "[bold green]🔧 Workflow[/bold green]",
            border_style = "green"
        ))

    def _handle_models(self):
        """Show all available Ollama models"""
        models = self.rag.list_available_models()

        if not models:
            print_error(
                "No models found!\n"
                "Run: ollama pull mistral"
            )
            return

        table = Table(title="🦙 Available Ollama Models")
        table.add_column("Model",   style="cyan")
        table.add_column("Active",  style="green")

        for model in models:
            is_active = (
                "✅ Active"
                if model == self.rag.model
                else ""
            )
            table.add_row(model, is_active)

        console.print(table)
        print_info(
            "Install new model: ollama pull <model_name>"
        )

    def _handle_switch(self):
        """Switch Ollama model"""
        model_name = Prompt.ask(
            "[cyan]Model name (e.g. llama3, mistral, phi3)[/cyan]"
        )
        self.rag.switch_model(model_name.strip())

    def _handle_stats(self):
        """Show system statistics"""
        stats = self.kb.get_stats()

        console.print(Panel(
            f"""
[bold]📊 System Statistics[/bold]
──────────────────────────────────────
Total Chunks    : {stats['total_chunks']:,}
Total Videos    : {stats['video_count']}
Embedding Model : {stats.get('embedding_model', 'N/A')}
Embedding Dims  : {stats.get('embedding_dims', 'N/A')}
LLM Model       : {self.rag.model}
Ollama URL      : {config.OLLAMA_BASE_URL}
Storage         : {stats['persist_dir']}
──────────────────────────────────────
Videos Indexed  : {', '.join(stats['videos']) or 'None'}
            """,
            border_style = "cyan"
        ))

    def _handle_quit(self):
        """Exit the application"""
        console.print(
            "[bold yellow]👋 Goodbye![/bold yellow]"
        )
        return "quit"

    # ══════════════════════════════════════════
    # 🎨 WELCOME SCREEN
    # ══════════════════════════════════════════

    def _print_welcome(self):
        """Print welcome screen"""
        console.print(Panel(
            f"""
[bold cyan]🧠 AI Video Learning Assistant[/bold cyan]
[bold green]100% Open Source & FREE![/bold green]

[bold]Embedding:[/bold]  {config.EMBEDDING_MODEL}
[bold]LLM:[/bold]        {config.OLLAMA_MODEL} (via Ollama)

[bold]Commands:[/bold]
  [cyan]research[/cyan] → Auto find & ingest by topic  ⭐ NEW
  [cyan]add[/cyan]      → Add single YouTube video
  [cyan]ask[/cyan]      → Ask a question (full answer)
  [cyan]stream[/cyan]   → Ask a question (live typing)
  [cyan]list[/cyan]     → List all indexed videos
  [cyan]queue[/cyan]    → Show ingestion queue status  ⭐ NEW
  [cyan]retry[/cyan]    → Retry failed queue items     ⭐ NEW
  [cyan]summary[/cyan]  → Summarize a video
  [cyan]workflow[/cyan] → Extract steps from video
  [cyan]models[/cyan]   → List Ollama models
  [cyan]switch[/cyan]   → Switch LLM model
  [cyan]stats[/cyan]    → System statistics
  [cyan]quit[/cyan]     → Exit
            """,
            border_style = "cyan"
        ))


# ══════════════════════════════════════════════
# ▶️ RUN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    app = AIVideoAssistant()
    app.run()