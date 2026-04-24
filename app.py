# app.py
# ─────────────────────────────────────────────
# RAG with Ollama (100% Free & Local!)
# ─────────────────────────────────────────────

import ollama
from config import config
from ingest import VideoIngester
from vector_store import KnowledgeBase
from utils import (
    print_header,
    print_success,
    print_error,
    print_info,
    console
)
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table


# ══════════════════════════════════════════════
# 🤖 OLLAMA RAG ENGINE.        
# ══════════════════════════════════════════════

class RAGEngine:
    """
    RAG Engine using Ollama (Local LLM)

    Flow:
    Question → Embed → Search → Context → Ollama → Answer
    All running locally! No API costs! 🎉
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb           = knowledge_base
        self.model        = config.OLLAMA_MODEL
        self.chat_history = []

    def ask(
        self,
        question: str,
        video_id:  str = None,
        n_context: int = 5 
    ) -> dict:
        """Main RAG pipeline"""

        print_info(f"🔍 Searching knowledge base...")

        # Step 1: Retrieve relevant chunks
        relevant_chunks = self.kb.search(
            query=question,
            n_results=n_context,
            video_id=video_id
        )

        if not relevant_chunks:
            return {
                "question":    question,
                "answer":      (
                    "I couldn't find relevant information. "
                    "Please add videos first using 'add' command."
                ),
                "sources":     [],
                "context_used": 0,
                "model_used":  self.model
            }

        # Step 2: Build context
        context = self._build_context(relevant_chunks)

        # Step 3: Generate with Ollama (LOCAL!)
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

    def _generate_with_ollama(
        self,
        question: str,
        context:  str
    ) -> str:
        """Generate answer using local Ollama model"""

        prompt = f"""
{config.SYSTEM_PROMPT}

Context from video transcripts:
{context}

Question: {question}

Answer:"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,    # Factual
                    "top_p":       0.9,
                    "num_predict": 1000,   # Max tokens
                }
            )
            return response["response"]

        except Exception as e:
            print_error(f"Ollama error: {e}")
            return (
                f"Error generating answer: {e}\n"
                f"Make sure Ollama is running: ollama serve"
            )

    def stream_ask(
        self,
        question: str,
        video_id:  str = None
    ):
        """
        Stream answer token by token
        Better user experience for long answers!
        """
        relevant_chunks = self.kb.search(
            query=question,
            n_results=5,
            video_id=video_id
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

        # Stream tokens as they generate
        for chunk in ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,         # Stream mode!
            options={"temperature": 0.3}
        ):
            print(chunk["response"], end="", flush=True)

        print("\n")

    def summarize_video(self, video_id: str) -> str:
        """Summarize entire video"""
        return self.ask(
            question=(
                "Give me a comprehensive summary. "
                "What are the main topics, key points, "
                "and important takeaways from this video?"
            ),
            video_id=video_id,
            n_context=10
        )["answer"]

    def extract_workflow(self, video_id: str) -> str:
        """Extract workflows and tutorials"""
        return self.ask(
            question=(
                "Extract step-by-step workflows, processes, "
                "or tutorials. Format as numbered steps."
            ),
            video_id=video_id,
            n_context=8
        )["answer"]

    def list_available_models(self) -> list[str]:
        """List all Ollama models available"""
        try:
            models = ollama.list()
            return [
                m['name']
                for m in models.get('models', [])
            ]
        except Exception:
            return []

    def switch_model(self, model_name: str):
        """Switch to different Ollama model"""
        available = self.list_available_models()
        model_base_names = [
            m.split(':')[0] for m in available
        ]

        if model_name not in model_base_names:
            print_error(
                f"Model '{model_name}' not found!\n"
                f"Available: {model_base_names}\n"
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

        commands = [
            "add", "ask", "stream",
            "list", "summary", "workflow",
            "models", "switch", "stats", "quit"
        ]

        while True:
            try:
                command = Prompt.ask(
                    "\n[bold cyan]Command[/bold cyan]",
                    choices=commands,
                    default="ask"
                )

                handlers = {
                    "add":      self._handle_add,
                    "ask":      self._handle_ask,
                    "stream":   self._handle_stream,
                    "list":     self._handle_list,
                    "summary":  self._handle_summary,
                    "workflow": self._handle_workflow,
                    "models":   self._handle_models,
                    "switch":   self._handle_switch,
                    "stats":    self._handle_stats,
                    "quit":     self._handle_quit
                }

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

    def _handle_add(self):
        """Add YouTube video"""
        url = Prompt.ask("[cyan]YouTube URL[/cyan]")
        result = self.ingester.ingest(url)
        if result["status"] == "success":
            self.kb.add_video(result)

    def _handle_ask(self):
        """Ask question (full response)"""
        question = Prompt.ask("[cyan]Your question[/cyan]")
        result = self.rag.ask(question)

        console.print(Panel(
            f"[white]{result['answer']}[/white]",
            title=(
                f"[bold green]🤖 {result['model_used']}[/bold green]"
            ),
            border_style="green"
        ))
        print_info(
            f"Used {result['context_used']} video excerpts"
        )

    def _handle_stream(self):
        """Ask question with streaming response"""
        question = Prompt.ask("[cyan]Your question[/cyan]")
        self.rag.stream_ask(question)

    def _handle_list(self):
        """List all indexed videos"""
        videos = self.kb.store.list_videos()

        if not videos:
            print_info("No videos yet! Use 'add' command.")
            return

        table = Table(title="📹 Indexed Videos")
        table.add_column("Video ID", style="cyan")
        table.add_column("Status",   style="green")

        for video in videos:
            table.add_row(video, "✅ Indexed")

        console.print(table)

    def _handle_summary(self):
        """Summarize a video"""
        videos = self.kb.store.list_videos()
        if not videos:
            print_info("No videos indexed yet!")
            return

        video_id = Prompt.ask(
            "[cyan]Video ID[/cyan]",
            choices=videos
        )
        print_info("Generating summary...")
        summary = self.rag.summarize_video(video_id)
        console.print(Panel(
            f"[white]{summary}[/white]",
            title=f"[bold green]📋 Summary[/bold green]",
            border_style="green"
        ))

    def _handle_workflow(self):
        """Extract workflow from video"""
        videos = self.kb.store.list_videos()
        if not videos:
            print_info("No videos indexed yet!")
            return

        video_id = Prompt.ask(
            "[cyan]Video ID[/cyan]",
            choices=videos
        )
        workflow = self.rag.extract_workflow(video_id)
        console.print(Panel(
            f"[white]{workflow}[/white]",
            title="[bold green]🔧 Workflow[/bold green]",
            border_style="green"
        ))

    def _handle_models(self):
        """Show available Ollama models"""
        models = self.rag.list_available_models()

        table = Table(title="🦙 Available Ollama Models")
        table.add_column("Model",   style="cyan")
        table.add_column("Current", style="green")

        for model in models:
            is_current = "✅ Active" if (
                model.split(':')[0] == self.rag.model
            ) else ""
            table.add_row(model, is_current)

        console.print(table)
        print_info(
            "Install new model: ollama pull <model_name>"
        )

    def _handle_switch(self):
        """Switch Ollama model"""
        model_name = Prompt.ask(
            "[cyan]Model name (e.g. llama3, phi3)[/cyan]"
        )
        self.rag.switch_model(model_name)

    def _handle_stats(self):
        """Show statistics"""
        stats = self.kb.get_stats()
        console.print(Panel(
            f"""
[bold]📊 System Statistics[/bold]
──────────────────────────────────
Total Chunks    : {stats['total_chunks']:,}
Total Videos    : {stats['video_count']}
Embedding Model : {stats['embedding_model']}
Embedding Dims  : {stats['embedding_dims']}
LLM Model       : {self.rag.model}
Ollama URL      : {config.OLLAMA_BASE_URL}
Storage         : {stats['persist_dir']}
──────────────────────────────────
Videos: {', '.join(stats['videos']) or 'None'}
            """,
            border_style="cyan"
        ))

    def _handle_quit(self):
        """Exit application"""
        console.print("[bold yellow]👋 Goodbye![/bold yellow]")
        return "quit"

    def _print_welcome(self):
        """Welcome screen"""
        console.print(Panel(
            f"""
[bold cyan]🧠 AI Video Learning Assistant[/bold cyan]
[bold green]100% Open Source & FREE![/bold green]

[bold]Embedding:[/bold]  {config.EMBEDDING_MODEL}
[bold]LLM:[/bold]        {config.OLLAMA_MODEL} (via Ollama)

[bold]Commands:[/bold]
  [cyan]add[/cyan]      → Add YouTube video
  [cyan]ask[/cyan]      → Ask question (full)
  [cyan]stream[/cyan]   → Ask question (streaming)
  [cyan]list[/cyan]     → List videos
  [cyan]summary[/cyan]  → Summarize video
  [cyan]workflow[/cyan] → Extract workflow
  [cyan]models[/cyan]   → List Ollama models
  [cyan]switch[/cyan]   → Switch LLM model
  [cyan]stats[/cyan]    → System statistics
  [cyan]quit[/cyan]     → Exit
            """,
            border_style="cyan"
        ))


# ══════════════════════════════════════════════
# ▶️ RUN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    app = AI_VIDEO_ASSISTANT = AIVideoAssistant()
    app.run() 