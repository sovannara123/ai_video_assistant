# config.py
# ─────────────────────────────────────────────
# Fixed version — handles new Ollama API structure
# ─────────────────────────────────────────────

import os
import requests
from dotenv import load_dotenv

load_dotenv()


class Config:

    # ══════════════════════════════════════════
    # 🔢 EMBEDDING MODEL
    # ══════════════════════════════════════════
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2"
    )

    # ══════════════════════════════════════════
    # 🦙 OLLAMA SETTINGS
    # ══════════════════════════════════════════
    OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "mistral")
    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    # ══════════════════════════════════════════
    # ✂️ CHUNKING
    # ══════════════════════════════════════════
    CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # ══════════════════════════════════════════
    # 🔍 RETRIEVAL
    # ══════════════════════════════════════════
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", 5))

    # ══════════════════════════════════════════
    # 💾 PATHS
    # ══════════════════════════════════════════
    DATA_DIR       = "data"
    TRANSCRIPT_DIR = "data/transcripts"
    CHUNKS_DIR     = "data/chunks"
    CHROMA_DIR     = "data/chroma_db"

    # ══════════════════════════════════════════
    # 🤖 SYSTEM PROMPT
    # ══════════════════════════════════════════
    SYSTEM_PROMPT = """
You are an AI Learning Assistant that helps
users learn from YouTube video content.

Your rules:
- Answer ONLY based on provided context
- Be clear, structured, and educational
- Use bullet points and examples
- If answer not in context, say honestly:
  "I don't find this in the indexed videos"
- Keep answers focused and concise

Always structure your answer as:
1. Direct answer
2. Key details
3. Example (if available in context)
    """

    # ══════════════════════════════════════════
    # 🔧 HELPER: Get model names safely
    # ══════════════════════════════════════════
    @classmethod
    def _get_ollama_models(cls) -> list[str]:
        """
        Safely extract model names from Ollama
        Handles different Ollama API versions
        """
        import ollama

        try:
            response = ollama.list()
            
            # ── Try different response structures ──

            # Structure 1: response is dict with 'models' key
            # {'models': [{'name': 'mistral:latest', ...}]}
            if isinstance(response, dict):
                models_list = response.get('models', [])

                if models_list:
                    first = models_list[0]

                    # Sub-structure A: dict with 'name'
                    if isinstance(first, dict):
                        if 'name' in first:
                            return [
                                m['name'].split(':')[0]
                                for m in models_list
                            ]
                        # Sub-structure B: dict with 'model'
                        elif 'model' in first:
                            return [
                                m['model'].split(':')[0]
                                for m in models_list
                            ]

            # Structure 2: response has .models attribute
            # (Pydantic model from newer ollama library)
            if hasattr(response, 'models'):
                models_list = response.models

                if models_list:
                    first = models_list[0]

                    # Pydantic object with .name attribute
                    if hasattr(first, 'name'):
                        return [
                            m.name.split(':')[0]
                            for m in models_list
                        ]
                    # Pydantic object with .model attribute
                    elif hasattr(first, 'model'):
                        return [
                            m.model.split(':')[0]
                            for m in models_list
                        ]

            # Structure 3: Use REST API directly (most reliable!)
            return cls._get_models_via_rest()

        except Exception as e:
            print(f"⚠️  ollama.list() failed: {e}")
            # Fallback to REST API
            return cls._get_models_via_rest()

    @classmethod
    def _get_models_via_rest(cls) -> list[str]:
        """
        Get models directly via REST API
        Most reliable across all Ollama versions
        """
        try:
            response = requests.get(
                f"{cls.OLLAMA_BASE_URL}/api/tags",
                timeout=5
            )

            if response.status_code == 200:
                data   = response.json()
                models = data.get('models', [])
                return [
                    m['name'].split(':')[0]
                    for m in models
                    if 'name' in m
                ]
            return []

        except Exception as e:
            print(f"⚠️  REST API failed: {e}")
            return []

    # ══════════════════════════════════════════
    # ✅ VALIDATE
    # ══════════════════════════════════════════
    @classmethod
    def validate(cls):
        """
        Validate all systems are ready
        Fixed to handle all Ollama API versions
        """
        print("\n🔍 Validating configuration...\n")
        errors = []

        # ── Check 1: Ollama server running ─────
        print("Checking Ollama server...")
        try:
            response = requests.get(
                f"{cls.OLLAMA_BASE_URL}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                print("✅ Ollama server is running!")
            else:
                errors.append(
                    f"❌ Ollama returned status: "
                    f"{response.status_code}"
                )
        except requests.exceptions.ConnectionError:
            errors.append(
                "❌ Ollama server not running!\n"
                "   Fix: open new terminal and run:\n"
                "   → ollama serve"
            )
            # Cannot continue without server
            print("\n".join(errors))
            raise RuntimeError(
                "Ollama server not running! "
                "Run: ollama serve"
            )

        # ── Check 2: Model available ───────────
        print(f"Checking model '{cls.OLLAMA_MODEL}'...")
        try:
            available_models = cls._get_ollama_models()
            print(f"   Found models: {available_models}")

            if not available_models:
                errors.append(
                    f"⚠️  No models found!\n"
                    f"   Fix: ollama pull {cls.OLLAMA_MODEL}"
                )
            elif cls.OLLAMA_MODEL not in available_models:
                errors.append(
                    f"❌ Model '{cls.OLLAMA_MODEL}' not installed!\n"
                    f"   Available: {available_models}\n"
                    f"   Fix: ollama pull {cls.OLLAMA_MODEL}"
                )
            else:
                print(f"✅ Model '{cls.OLLAMA_MODEL}' is ready!")

        except Exception as e:
            errors.append(f"❌ Model check failed: {e}")

        # ── Check 3: sentence-transformers ─────
        print("Checking sentence-transformers...")
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ sentence-transformers installed!")
        except ImportError:
            errors.append(
                "❌ sentence-transformers missing!\n"
                "   Fix: pip install sentence-transformers"
            )

        # ── Check 4: ChromaDB ──────────────────
        print("Checking ChromaDB...")
        try:
            import chromadb
            print("✅ ChromaDB installed!")
        except ImportError:
            errors.append(
                "❌ ChromaDB missing!\n"
                "   Fix: pip install chromadb"
            )

        # ── Check 5: YouTube API ───────────────
        print("Checking YouTube transcript API...")
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            print("✅ YouTube transcript API installed!")
        except ImportError:
            errors.append(
                "❌ YouTube transcript API missing!\n"
                "   Fix: pip install youtube-transcript-api"
            )

        # ── Results ────────────────────────────
        print()
        if errors:
            print("❌ Issues found:")
            for error in errors:
                print(f"\n{error}")
            raise RuntimeError(
                "\nConfiguration validation failed!\n"
                "Fix the issues above and try again."
            )

        print("✅ All systems ready! 🚀")
        cls.show_config()
        return True

    # ══════════════════════════════════════════
    # 📊 SHOW CONFIG
    # ══════════════════════════════════════════
    @classmethod
    def show_config(cls):
        """Display current configuration"""
        available = cls._get_ollama_models()
        print(f"""
╔══════════════════════════════════════════════╗
  ⚙️  Current Configuration
╠══════════════════════════════════════════════╣
  🔢 Embedding   : {cls.EMBEDDING_MODEL}
  🦙 LLM Model   : {cls.OLLAMA_MODEL}
  🌐 Ollama URL  : {cls.OLLAMA_BASE_URL}
  ✂️  Chunk Size  : {cls.CHUNK_SIZE} words
  🔍 Max Results : {cls.MAX_RESULTS}
  💾 Storage     : {cls.CHROMA_DIR}
╠══════════════════════════════════════════════╣
  📦 Available Models: {available}
╚══════════════════════════════════════════════╝
        """)


config = Config()