"""
Hybrid Retrieval Module (BM25 + FAISS) for corpus-grounded responses.
Indexes all corpus documents and retrieves relevant chunks.
"""

import os
import re
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from models import RetrievedDocument
from config import (
    DATA_DIR, REPO_ROOT, CORPUS_DIRS,
    BM25_TOP_K, FAISS_TOP_K, FINAL_TOP_K,
    RANDOM_SEED,
)

# Lazy imports for heavy dependencies
_sentence_model = None
_faiss_index = None


def _get_sentence_model():
    """Lazy-load the sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


class CorpusRetriever:
    """
    Hybrid retriever combining BM25 (keyword) and FAISS (semantic) search.
    Ensures deterministic retrieval with fixed seeds.
    """

    def __init__(self):
        self.documents: list[dict] = []  # {file_path, content, tokens}
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index = None
        self.embeddings: Optional[np.ndarray] = None
        self._indexed = False

        # Set for validating file paths
        self.valid_paths: set[str] = set()

    def index_corpus(self) -> None:
        """
        Index all markdown files in the data directory.
        Called once at startup.
        """
        if self._indexed:
            return

        print("[Retriever] Indexing corpus...")
        self.documents = []

        # Walk all data directories and collect markdown files
        for root, dirs, files in os.walk(DATA_DIR):
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

        # Build BM25 index
        tokenized = [doc["tokens"] for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

        # Build FAISS index
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

    def _build_faiss_index(self) -> None:
        """Build FAISS vector index from document embeddings."""
        import faiss

        model = _get_sentence_model()
        texts = [doc["content"][:500] for doc in self.documents]  # limit for speed

        print("[Retriever] Computing embeddings...")
        self.embeddings = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,
        )

        # Build FAISS index (Inner Product for cosine similarity with normalized vectors)
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings.astype(np.float32))
        print(f"[Retriever] FAISS index built with {self.faiss_index.ntotal} vectors")

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        company_filter: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """
        Hybrid retrieval: BM25 + FAISS, merged and de-duplicated.

        Args:
            query: Search query text
            top_k: Number of results to return
            company_filter: Optional company to prioritize (devplatform/claude/visa)

        Returns:
            List of RetrievedDocument, sorted by relevance
        """
        if not self._indexed:
            self.index_corpus()

        if not query or not query.strip():
            return []

        # BM25 retrieval
        bm25_results = self._bm25_search(query, BM25_TOP_K)

        # FAISS retrieval
        faiss_results = self._faiss_search(query, FAISS_TOP_K)

        # Merge and de-duplicate
        merged = self._merge_results(bm25_results, faiss_results, company_filter)

        # Return top-k
        return merged[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedDocument]:
        """BM25 keyword search."""
        tokens = self._tokenize(query)
        if not tokens:
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
        """FAISS semantic search."""
        model = _get_sentence_model()
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.faiss_index.search(query_embedding, top_k)

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
        Merge BM25 and FAISS results using Reciprocal Rank Fusion (RRF).
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

        # De-duplicate by file path (keep highest-scoring chunk per file)
        seen_files: set[str] = set()
        results = []
        for key in sorted_keys:
            doc = doc_map[key]
            if doc.file_path not in seen_files:
                doc.score = doc_scores[key]
                results.append(doc)
                seen_files.add(doc.file_path)

        return results

    def validate_file_path(self, path: str) -> bool:
        """Check if a file path exists in the corpus."""
        # Normalize path
        normalized = path.replace("\\", "/").strip()
        return normalized in self.valid_paths

    def get_all_valid_paths(self) -> set[str]:
        """Return all valid corpus file paths."""
        return self.valid_paths.copy()
