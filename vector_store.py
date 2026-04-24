# vector_store.py
# ─────────────────────────────────────────────
# Open Source Embeddings + ChromaDB
# sentence-transformers replaces OpenAI!
# ─────────────────────────────────────────────

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from config import config
from utils import (
    print_header,
    print_success,
    print_error,
    print_info,
    console
)


# ══════════════════════════════════════════════
# 🔢 OPEN SOURCE EMBEDDING ENGINE
# ══════════════════════════════════════════════

class EmbeddingEngine:
    """
    FREE Local Embeddings using sentence-transformers

    Models comparison:
    ──────────────────────────────────────────────
    Model                      Size   Speed  Quality
    ──────────────────────────────────────────────
    all-MiniLM-L6-v2          80MB   Fast   Good
    all-mpnet-base-v2         420MB  Medium Great
    BAAI/bge-small-en-v1.5   130MB  Fast   Great
    BAAI/bge-large-en-v1.5   1.3GB  Slow   Best
    multi-qa-MiniLM-L6-cos-v1 80MB  Fast   Good QA
    ──────────────────────────────────────────────
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL

        print_info(
            f"Loading embedding model: {self.model_name}"
        )
        print_info("(First run downloads model, then cached)")

        # Load model (downloads if not cached)
        self.model = SentenceTransformer(self.model_name)
        self.embed_count = 0

        # Get embedding dimensions
        test_embed = self.model.encode("test")
        self.dimensions = len(test_embed)

        print_success(
            f"Embedding model ready!\n"
            f"  Model: {self.model_name}\n"
            f"  Dimensions: {self.dimensions}\n"
            f"  Device: {self.model.device}"
        )

    def embed_text(self, text: str) -> list[float]:
        """Convert single text to embedding vector"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,   # Normalize for cosine
            show_progress_bar=False
        )
        self.embed_count += 1
        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> list[list[float]]:
        """
        Convert multiple texts to embeddings

        sentence-transformers handles batching internally!
        Much faster than calling one by one
        """
        print_info(
            f"Embedding {len(texts)} texts... "
            f"(model: {self.model_name})"
        )

        # Encode all at once with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # For cosine similarity
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.embed_count += len(texts)

        print_success(
            f"Created {len(embeddings)} embeddings! "
            f"Shape: {embeddings.shape}"
        )

        return embeddings.tolist()

    def similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate similarity between two texts
        Returns: 0.0 (different) to 1.0 (identical)
        """
        emb1 = np.array(self.embed_text(text1))
        emb2 = np.array(self.embed_text(text2))

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        return float(similarity)

    def get_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "device": str(self.model.device),
            "total_embedded": self.embed_count
        }


# ══════════════════════════════════════════════
# 🗄️ VECTOR DATABASE (ChromaDB - unchanged)
# ══════════════════════════════════════════════

class VectorStore:
    """
    ChromaDB vector database
    Works the same with open source embeddings!
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or config.CHROMA_DIR
        self.collection_name = "video_knowledge"

        self.client = chromadb.PersistentClient(
            path=self.persist_dir
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print_success(
            f"ChromaDB ready! "
            f"Entries: {self.collection.count()}"
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ):
        """Store chunks with embeddings"""
        if not chunks:
            print_error("No chunks to add!")
            return

        ids       = [c["id"] for c in chunks]
        texts     = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Check for existing
        existing     = self.collection.get(ids=ids)
        existing_ids = set(existing["ids"])
        new_indices  = [
            i for i, id_ in enumerate(ids)
            if id_ not in existing_ids
        ]

        if not new_indices:
            print_info("All chunks already indexed!")
            return

        # Add only new
        self.collection.add(
            ids=[ids[i] for i in new_indices],
            documents=[texts[i] for i in new_indices],
            embeddings=[embeddings[i] for i in new_indices],
            metadatas=[metadatas[i] for i in new_indices]
        )

        print_success(
            f"Added {len(new_indices)} chunks! "
            f"({len(existing_ids)} already existed)"
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        video_id: str = None
    ) -> list[dict]:
        """Find most similar chunks"""
        n_results = min(n_results, self.collection.count())

        if n_results == 0:
            return []

        where_filter = None
        if video_id:
            where_filter = {"video_id": {"$eq": video_id}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id":         results["ids"][0][i],
                "text":       results["documents"][0][i],
                "metadata":   results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]
            })

        return formatted

    def list_videos(self) -> list[str]:
        """List all indexed video IDs"""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])
        return list(set(
            m["video_id"] for m in results["metadatas"]
        ))

    def delete_video(self, video_id: str):
        """Delete all chunks from a video"""
        self.collection.delete(
            where={"video_id": {"$eq": video_id}}
        )
        print_success(f"Deleted video: {video_id}")

    def get_stats(self) -> dict:
        """Database statistics"""
        return {
            "total_chunks":    self.collection.count(),
            "collection_name": self.collection_name,
            "persist_dir":     self.persist_dir
        }


# ══════════════════════════════════════════════
# 🧠 KNOWLEDGE BASE (Combines Both)
# ══════════════════════════════════════════════

class KnowledgeBase:
    """
    Complete Knowledge Base
    EmbeddingEngine + VectorStore combined
    """

    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.store    = VectorStore()

    def add_video(self, ingestion_result: dict) -> bool:
        """Add ingested video to knowledge base"""
        print_header("🧠 Adding to Knowledge Base")

        if ingestion_result["status"] != "success":
            print_error("Cannot add failed ingestion")
            return False

        chunks = ingestion_result["chunks"]
        texts  = [chunk["text"] for chunk in chunks]

        # Create embeddings (FREE & LOCAL!)
        embeddings = self.embedder.embed_batch(texts)

        # Store in ChromaDB
        self.store.add_chunks(chunks, embeddings)

        print_success(
            f"✅ Video added! "
            f"Video: {ingestion_result['video_id']}"
        )
        return True

    def search(
        self,
        query: str,
        n_results: int = 5,
        video_id: str = None
    ) -> list[dict]:
        """Search with natural language query"""

        # Embed query using same local model
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        return self.store.search(
            query_embedding,
            n_results=n_results,
            video_id=video_id
        )

    def get_stats(self) -> dict:
        """Full statistics"""
        stats           = self.store.get_stats()
        stats["videos"] = self.store.list_videos()
        stats["video_count"] = len(stats["videos"])
        stats["embedding_model"] = self.embedder.model_name
        stats["embedding_dims"]  = self.embedder.dimensions
        return stats