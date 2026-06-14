import os
import sys
import json
import pickle
from typing import Optional, Union
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBED_MODEL, TOP_K, IMAGE_THRESHOLD, CACHE_DIR, DB_DIR


class Retriever:

    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBED_MODEL)
        self.text_index: Optional[faiss.Index] = None
        self.image_index: Optional[faiss.Index] = None
        self.text_metadata: list[dict] = []
        self.image_metadata: list[dict] = []
        self.cache_dir = CACHE_DIR
        self.vector_db_dir = DB_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)

        # merged index paths (used for search)
        self._text_index_path = os.path.join(self.vector_db_dir, "text.index")
        self._text_meta_path = os.path.join(self.vector_db_dir, "text_meta.pkl")
        self._image_index_path = os.path.join(self.vector_db_dir, "image.index")
        self._image_meta_path = os.path.join(self.vector_db_dir, "image_meta.pkl")
        self._merged_stamp_path = os.path.join(self.vector_db_dir, "merged_stamp.json")

        # per-pdf cache directory
        self._per_pdf_dir = os.path.join(self.vector_db_dir, "per_pdf")
        os.makedirs(self._per_pdf_dir, exist_ok=True)

    def __call__(self, query: str) -> tuple[list[dict], list[dict]]:
        """Search both text and image indexes. Returns (text_results, image_results)."""
        query_vec = self._embed([query])

        text_results = self._search(
            self.text_index, query_vec, self.text_metadata, TOP_K
        )
        image_results = self._search(
            self.image_index, query_vec, self.image_metadata, TOP_K,
            threshold=IMAGE_THRESHOLD
        )

        return text_results, image_results

    # ── merged index: load / save ──

    def load_merged_if_valid(self, pdf_paths: Union[str, list[str]]) -> bool:
        """Load merged FAISS indexes if they exist and ALL pdfs are unchanged."""
        pdf_paths = self._normalize_paths(pdf_paths)

        if not self._merged_stamp_valid(pdf_paths):
            return False

        try:
            self.text_index = faiss.read_index(self._text_index_path)
            with open(self._text_meta_path, "rb") as f:
                self.text_metadata = pickle.load(f)

            if os.path.exists(self._image_index_path):
                self.image_index = faiss.read_index(self._image_index_path)
                with open(self._image_meta_path, "rb") as f:
                    self.image_metadata = pickle.load(f)

            print(f"  -> Merged index loaded: {self.text_index.ntotal} text vectors")
            if self.image_index:
                print(f"  -> Merged index loaded: {self.image_index.ntotal} image vectors")
            return True

        except Exception as e:
            print(f"  -> Merged index load failed: {e}")
            return False

    def save_merged_stamp(self, pdf_paths: Union[str, list[str]]) -> None:
        """Save stamp for the merged index tracking all pdf mtimes."""
        pdf_paths = self._normalize_paths(pdf_paths)
        with open(self._merged_stamp_path, "w") as f:
            json.dump({
                "pdfs": {os.path.abspath(p): os.path.getmtime(p) for p in pdf_paths}
            }, f)

    def _merged_stamp_valid(self, pdf_paths: list[str]) -> bool:
        """Check if merged index stamp matches current set of pdfs."""
        required = [self._text_index_path, self._text_meta_path, self._merged_stamp_path]
        if not all(os.path.exists(p) for p in required):
            return False
        try:
            with open(self._merged_stamp_path) as f:
                stamp = json.load(f)
            current = {os.path.abspath(p): os.path.getmtime(p) for p in pdf_paths}
            return stamp.get("pdfs") == current
        except Exception:
            return False

    # ── per-pdf cache: load / save ──

    def _pdf_cache_dir(self, pdf_path: str) -> str:
        """Return the per-pdf cache directory path."""
        pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
        return os.path.join(self._per_pdf_dir, pdf_stem)

    def load_pdf_chunks(self, pdf_path: str) -> Optional[tuple[list[dict], list[dict]]]:
        """Load cached chunks for a single pdf if it hasn't changed.
        Returns (text_chunks, image_chunks) or None if cache invalid."""
        pdf_dir = self._pdf_cache_dir(pdf_path)
        stamp_path = os.path.join(pdf_dir, "stamp.json")
        chunks_path = os.path.join(pdf_dir, "chunks.pkl")

        if not os.path.exists(stamp_path) or not os.path.exists(chunks_path):
            return None

        try:
            with open(stamp_path) as f:
                stamp = json.load(f)
            if stamp.get("mtime") != os.path.getmtime(pdf_path):
                return None

            with open(chunks_path, "rb") as f:
                data = pickle.load(f)
            return data["text_chunks"], data["image_chunks"]

        except Exception:
            return None

    def save_pdf_chunks(self, pdf_path: str, text_chunks: list[dict], image_chunks: list[dict]) -> None:
        """Save chunks for a single pdf to its own cache directory."""
        pdf_dir = self._pdf_cache_dir(pdf_path)
        os.makedirs(pdf_dir, exist_ok=True)

        stamp_path = os.path.join(pdf_dir, "stamp.json")
        chunks_path = os.path.join(pdf_dir, "chunks.pkl")

        with open(stamp_path, "w") as f:
            json.dump({"mtime": os.path.getmtime(pdf_path)}, f)

        with open(chunks_path, "wb") as f:
            pickle.dump({"text_chunks": text_chunks, "image_chunks": image_chunks}, f)

    # ── build merged FAISS indexes ──

    def index_text(self, chunks: list[dict]) -> None:
        """Build merged FAISS index from text chunks and persist to disk."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        dimension = embeddings.shape[1]
        self.text_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.text_index.add(embeddings)
        self.text_metadata = chunks

        faiss.write_index(self.text_index, self._text_index_path)
        with open(self._text_meta_path, "wb") as f:
            pickle.dump(self.text_metadata, f)

    def index_images(self, chunks: list[dict]) -> None:
        """Build merged FAISS index from image chunks and persist to disk."""
        valid = [c for c in chunks if self._build_image_text(c)]
        if not valid:
            for p in (self._image_index_path, self._image_meta_path):
                if os.path.exists(p):
                    os.remove(p)
            return

        texts = [self._build_image_text(c) for c in valid]
        embeddings = self._embed(texts)

        dimension = embeddings.shape[1]
        self.image_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.image_index.add(embeddings)
        self.image_metadata = valid

        faiss.write_index(self.image_index, self._image_index_path)
        with open(self._image_meta_path, "wb") as f:
            pickle.dump(self.image_metadata, f)

    # ── helpers ──

    def _normalize_paths(self, pdf_paths: Union[str, list[str]]) -> list[str]:
        """Ensure pdf_paths is always a list."""
        if isinstance(pdf_paths, str):
            return [pdf_paths]
        return pdf_paths

    def _build_image_text(self, chunk: dict) -> str:
        """Combine caption and description into one searchable string."""
        parts = []
        if chunk.get("caption"):
            parts.append(chunk["caption"])
        if chunk.get("description"):
            parts.append(chunk["description"])
        return " ".join(parts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype("float32")

    def _search(
        self,
        index: Optional[faiss.Index],
        query_vec: np.ndarray,
        metadata: list[dict],
        top_k: int,
        threshold: Optional[float] = None
    ) -> list[dict]:
        """Search a FAISS index and return ranked results."""
        if index is None or index.ntotal == 0:
            return []

        faiss.normalize_L2(query_vec)
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if threshold is not None and score < threshold:
                continue

            result = metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

        return results