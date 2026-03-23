

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class SearchResult:
    """A document chunk returned by semantic search."""
    text: str
    score: float        # Similarity score (higher = more similar)
    chunk_index: int    # Position in original document
    metadata: dict = field(default_factory=dict)


class DocumentEmbedder:
    
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: Optional[str] = None):
        """
        LEARNING NOTE: sentence-transformers wraps Hugging Face models with
        a simpler API for generating sentence/document embeddings.
        """
        model_name = model_name or self.DEFAULT_MODEL
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: list = []  # Store original text chunks for retrieval

    def index_document(self, text: str, chunk_size: int = 200, overlap: int = 50):
        
        self._chunks = self._create_chunks(text, chunk_size, overlap)
        print(f"Created {len(self._chunks)} chunks from document")

        print("Generating embeddings...")
        embeddings = self.model.encode(
            self._chunks,
            batch_size=32,
            show_progress_bar=len(self._chunks) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )

        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(embeddings.astype(np.float32))

        print(f"Index built: {self._index.ntotal} vectors")

    def search(self, query: str, top_k: int = 3) -> list:
        
        if self._index is None or self._index.ntotal == 0:
            print("WARNING: No document indexed yet. Call index_document() first.")
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores, indices = self._index.search(
            query_embedding.astype(np.float32),
            min(top_k, self._index.ntotal)
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for unfound results
                continue
            results.append(SearchResult(
                text=self._chunks[idx],
                score=float(score),
                chunk_index=int(idx),
            ))

        return results

    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
       
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Excerpt {i}]\n{result.text}")

        return "\n\n".join(context_parts)

    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> list:
      
        words = text.split()

        if len(words) <= chunk_size:
            return [text]  # Document is small enough to be one chunk

        chunks = []
        step = chunk_size - overlap  # How far to advance the window each time

        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 20:  # Skip very short chunks
                chunks.append(chunk)

            if end == len(words):
                break

        return chunks

if __name__ == "__main__":
    sample_doc = """
    Rechnung Nr. 2024-0042
    Datum: 15. März 2024
    
    Verkäufer: TechShop Berlin GmbH
    Adresse: Alexanderplatz 1, 10178 Berlin
    
    Käufer: Anna Schmidt
    Adresse: Unter den Linden 5, 10117 Berlin
    
    Artikel:
    - Dell XPS 15 Laptop (9530): 1.299,00 EUR
    - USB-C Hub 7-Port: 49,99 EUR
    - Laptop-Rucksack: 89,00 EUR
    
    Zwischensumme: 1.437,99 EUR
    MwSt. 19%: 273,22 EUR
    Gesamtbetrag: 1.711,21 EUR
    
    Zahlungsziel: 30 Tage
    IBAN: DE89 3704 0044 0532 0130 00
    Bank: Deutsche Bank AG
    """

    print("=== DocumentEmbedder Test ===\n")
    embedder = DocumentEmbedder()
    embedder.index_document(sample_doc, chunk_size=50, overlap=10)

    queries = [
        "What is the total price?",
        "Wie hoch ist der Gesamtbetrag?",   # German: "What is the total amount?"
        "Who is the buyer?",
        "What laptop was purchased?",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = embedder.search(query, top_k=2)
        for r in results:
            print(f"  [{r.score:.3f}] {r.text[:80]}...")
