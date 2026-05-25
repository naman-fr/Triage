"""
Hybrid Retrieval Module (BM25 + FAISS) with Voyage AI Embeddings & Cohere Reranking.
Indexes all corpus documents and retrieves relevant chunks.
Includes local caching of embeddings for speed and reproducibility.
Resilient to environment library failures (e.g. broken NumPy/SciPy installations).
"""

import os
import re
import pickle
import hashlib
import requests
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

from rank_bm25 import BM25Okapi

from models import RetrievedDocument
from config import (
    DATA_DIR, REPO_ROOT, CORPUS_DIRS,
    BM25_TOP_K, FAISS_TOP_K, FINAL_TOP_K,
    RANDOM_SEED, VOYAGE_API_KEY, COHERE_API_KEY,
)

# Lazy imports for heavy dependencies
_sentence_model = None
_sentence_model_failed = False


def _get_sentence_model():
    """Lazy-load the sentence transformer model with environment protection."""
    global _sentence_model, _sentence_model_failed
    if _sentence_model is None and not _sentence_model_failed:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[Retriever] Warning: SentenceTransformer model load failed ({e}). Fallback to BM25 search.")
            _sentence_model_failed = True
            _sentence_model = None
    return _sentence_model


class CorpusRetriever:
    """
    Hybrid retriever combining BM25 (keyword) and FAISS (semantic) search.
    Upgrades to Voyage AI embeddings and Cohere Reranking when API keys are present.
    Ensures deterministic retrieval with fixed seeds and local caching.
    """

    def __init__(self):
        self.documents: list[dict] = []  # {file_path, content, tokens}
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index = None
        self.embeddings: Optional[np.ndarray] = None
        self._indexed = False
        self.voyage_failed = False
        self.cohere_failed = False

        # Set for validating file paths
        self.valid_paths: set[str] = set()

        # Cache paths
        self.cache_dir = REPO_ROOT / "data" / ".cache"
        self.cache_dir.mkdir(exist_ok=True)

    def index_corpus(self) -> None:
        """
        Index all markdown files in the data directory.
        Called once at startup. Loads cached embeddings if available.
        """
        if self._indexed:
            return

        print("[Retriever] Indexing corpus...")
        self.documents = []

        # Walk all data directories and collect markdown files
        for root, dirs, files in os.walk(DATA_DIR):
            # Skip cache folder
            if ".cache" in root:
                continue

            for fname in sorted(files):  # sorted for determinism
                if not fname.endswith(".md"):
                    continue

                fpath = Path(root) / fname
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                # Compute relative path from repo root
                rel_path = str(fpath.relative_to(REPO_ROOT)).replace("\\", "/")
                self.valid_paths.add(rel_path)

                # Chunk the document if it's large
                chunks = self._chunk_document(content, rel_path)
                self.documents.extend(chunks)

        print(f"[Retriever] Indexed {len(self.documents)} chunks from {len(self.valid_paths)} files")

        if not self.documents:
            print("[Retriever] Warning: No documents found to index!")
            self._indexed = True
            return

        # Build BM25 index
        tokenized = [doc["tokens"] for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

        # Build FAISS index (inner product / cosine similarity)
        self._build_faiss_index()

        self._indexed = True

    def _chunk_document(
        self, content: str, file_path: str, chunk_size: int = 1500, overlap: int = 200
    ) -> list[dict]:
        """
        Split a document into overlapping chunks.
        Uses character-based chunking for simplicity and speed.
        """
        content = content.strip()
        if not content:
            return []

        # For small documents, don't chunk
        if len(content) <= chunk_size:
            tokens = self._tokenize(content)
            return [{"file_path": file_path, "content": content, "tokens": tokens}]

        chunks = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Try to break at a paragraph or sentence boundary
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind("\n\n", start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break
                else:
                    # Look for sentence break
                    sent_break = content.rfind(". ", start, end)
                    if sent_break > start + chunk_size // 2:
                        end = sent_break + 1

            chunk_text = content[start:end].strip()
            if chunk_text:
                tokens = self._tokenize(chunk_text)
                chunks.append({
                    "file_path": file_path,
                    "content": chunk_text,
                    "tokens": tokens,
                })

            start = end - overlap if end < len(content) else len(content)

        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer for BM25."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def _compute_voyage_embeddings(self, texts: list[str]) -> np.ndarray:
        """Call Voyage AI API to compute embeddings."""
        print(f"[Retriever] Computing {len(texts)} embeddings via Voyage AI...")
        headers = {
            "Authorization": f"Bearer {VOYAGE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        embeddings = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            body = {
                "input": batch,
                "model": "voyage-3"
            }
            try:
                res = requests.post(
                    "https://api.voyageai.com/v1/embeddings",
                    json=body,
                    headers=headers,
                    timeout=30
                )
                res.raise_for_status()
                data = res.json()
                embeddings.extend([item["embedding"] for item in data["data"]])
            except Exception as e:
                print(f"[Retriever] Voyage API error: {e}. Falling back to local SentenceTransformers.")
                raise RuntimeError("Voyage AI call failed") from e
                
        return np.array(embeddings, dtype=np.float32)

    def _build_faiss_index(self) -> None:
        """Build FAISS vector index, using local or Voyage embeddings (with cache)."""
        # Guard faiss imports for platform compliance
        try:
            import faiss
        except Exception as e:
            print(f"[Retriever] Warning: FAISS library failed to import ({e}). Vector search will be disabled.")
            self.faiss_index = None
            return

        # Unique identifier for the corpus content to validate cache
        corpus_hash = hashlib.md5(
            "".join(doc["content"] for doc in self.documents).encode("utf-8")
        ).hexdigest()

        use_voyage = bool(VOYAGE_API_KEY)
        cache_name = "voyage_cache.pkl" if use_voyage else "local_cache.pkl"
        cache_path = self.cache_dir / cache_name

        loaded_from_cache = False
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                if cache_data.get("hash") == corpus_hash:
                    self.embeddings = cache_data["embeddings"]
                    print(f"[Retriever] Loaded cached embeddings from {cache_name}")
                    loaded_from_cache = True
            except Exception as e:
                print(f"[Retriever] Failed to load cache: {e}")

        if not loaded_from_cache:
            texts = [doc["content"] for doc in self.documents]
            
            if use_voyage:
                try:
                    self.embeddings = self._compute_voyage_embeddings(texts)
                except Exception:
                    # Fall back to local
                    use_voyage = False
                    print("[Retriever] Reverting to local SentenceTransformers model...")
            
            if not use_voyage:
                # Use local sentence transformers
                model = _get_sentence_model()
                if model is None:
                    print("[Retriever] SentenceTransformers unavailable. FAISS index will be skipped.")
                    self.faiss_index = None
                    return

                try:
                    print("[Retriever] Computing embeddings via local SentenceTransformers...")
                    self.embeddings = model.encode(
                        texts,
                        show_progress_bar=False,
                        batch_size=64,
                        normalize_embeddings=True,
                    ).astype(np.float32)
                except Exception as e:
                    print(f"[Retriever] Embedding computation failed ({e}). Disabling vector search.")
                    self.faiss_index = None
                    return

            # Save to cache
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "hash": corpus_hash,
                        "embeddings": self.embeddings
                    }, f)
                print(f"[Retriever] Saved embeddings cache to {cache_name}")
            except Exception as e:
                print(f"[Retriever] Failed to save cache: {e}")

        try:
            # Build FAISS IndexFlatIP (Inner Product for Cosine Similarity on normalized vectors)
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            # Ensure normalization for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # prevent division by zero
            normalized_embeddings = self.embeddings / norms
            self.faiss_index.add(normalized_embeddings.astype(np.float32))
            print(f"[Retriever] FAISS index built with {self.faiss_index.ntotal} vectors (dim={dim})")
        except Exception as e:
            print(f"[Retriever] Error building FAISS index: {e}. Vector search disabled.")
            self.faiss_index = None

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        company_filter: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """
        Hybrid retrieval: BM25 + FAISS (Voyage or local), merged and optionally Cohere-reranked.

        Args:
            query: Search query text
            top_k: Number of results to return
            company_filter: Optional company to prioritize (devplatform/claude/visa)

        Returns:
            List of RetrievedDocument, sorted by relevance
        """
        if not self._indexed:
            self.index_corpus()

        if not self.documents or not query or not query.strip():
            return []

        # Retrieve a slightly larger pool for re-ranking/merging
        pool_size = max(top_k * 4, 20)

        # BM25 retrieval
        bm25_results = self._bm25_search(query, pool_size)

        # FAISS retrieval
        faiss_results = self._faiss_search(query, pool_size)

        # Merge results with RRF first
        merged = self._merge_results(bm25_results, faiss_results, company_filter)

        # Apply Cohere Reranking if API key is present
        if COHERE_API_KEY and not self.cohere_failed and merged:
            try:
                merged = self._cohere_rerank_pool(query, merged, top_k)
            except Exception as e:
                print(f"[Retriever] Cohere rerank failed: {e}. Blacklisting Cohere and falling back to standard RRF ranking.")
                self.cohere_failed = True
                merged = merged[:top_k]
        else:
            merged = merged[:top_k]

        return merged

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedDocument]:
        """BM25 keyword search."""
        tokens = self._tokenize(query)
        if not tokens or self.bm25 is None:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                results.append(RetrievedDocument(
                    file_path=doc["file_path"],
                    content=doc["content"],
                    score=float(scores[idx]),
                    source="bm25",
                ))
        return results

    def _faiss_search(self, query: str, top_k: int) -> list[RetrievedDocument]:
        """FAISS semantic search (supports Voyage or local embedding)."""
        if self.faiss_index is None:
            return []

        # Check if we are using Voyage for the index
        use_voyage = bool(VOYAGE_API_KEY) and not self.voyage_failed
        query_embedding = None

        if use_voyage:
            try:
                headers = {
                    "Authorization": f"Bearer {VOYAGE_API_KEY}",
                    "Content-Type": "application/json"
                }
                body = {
                    "input": [query],
                    "model": "voyage-3"
                }
                res = requests.post(
                    "https://api.voyageai.com/v1/embeddings",
                    json=body,
                    headers=headers,
                    timeout=10
                )
                res.raise_for_status()
                query_embedding = np.array(res.json()["data"][0]["embedding"], dtype=np.float32)
            except Exception as e:
                print(f"[Retriever] Voyage query embedding failed: {e}. Blacklisting Voyage and falling back to local SentenceTransformers.")
                self.voyage_failed = True
                use_voyage = False

        if not use_voyage or query_embedding is None:
            model = _get_sentence_model()
            if model is None:
                return []
            try:
                query_embedding = model.encode(
                    [query],
                    normalize_embeddings=True,
                ).astype(np.float32)[0]
            except Exception as e:
                print(f"[Retriever] local query embedding computation failed ({e}). skipping vector search.")
                return []

        # Ensure normalized query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        query_embedding = np.expand_dims(query_embedding, axis=0)
        try:
            scores, indices = self.faiss_index.search(query_embedding, top_k)
        except Exception as e:
            print(f"[Retriever] FAISS index search query failed ({e}).")
            return []

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0:
                doc = self.documents[idx]
                results.append(RetrievedDocument(
                    file_path=doc["file_path"],
                    content=doc["content"],
                    score=float(score),
                    source="faiss",
                ))
        return results

    def _merge_results(
        self,
        bm25_results: list[RetrievedDocument],
        faiss_results: list[RetrievedDocument],
        company_filter: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """
        Merge BM25 and FAISS results using Reciprocal Rank Fusion (RRF) with company boosting.
        """
        k = 60  # RRF constant

        # Compute RRF scores
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, RetrievedDocument] = {}

        for rank, doc in enumerate(bm25_results):
            key = f"{doc.file_path}:{hash(doc.content[:100])}"
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        for rank, doc in enumerate(faiss_results):
            key = f"{doc.file_path}:{hash(doc.content[:100])}"
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            if key not in doc_map:
                doc_map[key] = doc

        # Apply company filter boost
        if company_filter:
            company_lower = company_filter.lower()
            company_path_map = {
                "devplatform": "data/devplatform/",
                "claude": "data/claude/",
                "visa": "data/visa/",
            }
            boost_prefix = company_path_map.get(company_lower, "")
            if boost_prefix:
                for key in doc_scores:
                    doc = doc_map[key]
                    if doc.file_path.startswith(boost_prefix):
                        doc_scores[key] *= 1.5  # 50% boost for matching company

        # Sort by RRF score
        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)

        # De-duplicate by file path (keep highest-scoring chunk per file to maximize context coverage)
        seen_files: set[str] = set()
        results = []
        for key in sorted_keys:
            doc = doc_map[key]
            if doc.file_path not in seen_files:
                doc.score = doc_scores[key]
                results.append(doc)
                seen_files.add(doc.file_path)

        return results

    def _cohere_rerank_pool(
        self, query: str, documents: list[RetrievedDocument], top_k: int
    ) -> list[RetrievedDocument]:
        """Use Cohere Rerank API to reorder candidate documents."""
        print(f"[Retriever] Reranking {len(documents)} candidates via Cohere...")
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Format texts for Cohere
        doc_texts = [doc.content[:1500] for doc in documents]
        
        body = {
            "query": query,
            "documents": [{"text": t} for t in doc_texts],
            "model": "rerank-english-v3.0",
            "top_n": top_k
        }
        
        res = requests.post(
            "https://api.cohere.com/v1/rerank",
            json=body,
            headers=headers,
            timeout=15
        )
        res.raise_for_status()
        results_data = res.json()["results"]
        
        reranked = []
        for item in results_data:
            idx = item["index"]
            doc = documents[idx]
            doc.score = float(item["relevance_score"])
            doc.source = "cohere_rerank"
            reranked.append(doc)
            
        return reranked

    def validate_file_path(self, path: str) -> bool:
        """Check if a file path exists in the corpus."""
        normalized = path.replace("\\", "/").strip()
        return normalized in self.valid_paths

    def get_all_valid_paths(self) -> set[str]:
        """Return all valid corpus file paths."""
        return self.valid_paths.copy()
