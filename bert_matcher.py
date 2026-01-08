"""
BERT-based Vector Matcher for Ontology Matching.
Uses sentence-transformers to create embeddings and find similar concepts.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class VectorIndex:
    """
    Vector-based index for semantic similarity search.
    Uses BERT embeddings to find similar concepts.
    """

    def __init__(self, df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the VectorIndex.

        Args:
            df: DataFrame with columns 'id', 'label', 'context'
            model_name: Name of the sentence-transformers model to use
        """
        self.df = df
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.texts = None

    def build_index(self) -> None:
        """
        Build the vector index by generating embeddings for all concepts.
        Combines label and context for better semantic representation.
        """
        print(f"Building vector index with model '{self.model_name}'...")

        # Combine label and context for richer embeddings
        self.texts = []
        for _, row in self.df.iterrows():
            label = row.get('label', '')
            context = row.get('context', '')

            # Combine label and context (limit context to avoid too long texts)
            if label and context:
                # Truncate context to ~500 chars to keep it manageable
                context_truncated = context[:500] if len(context) > 500 else context
                combined_text = f"{label}. {context_truncated}"
            elif label:
                combined_text = label
            elif context:
                combined_text = context[:500] if len(context) > 500 else context
            else:
                combined_text = ""

            self.texts.append(combined_text)

        # Generate embeddings for all texts
        print(f"Generating embeddings for {len(self.texts)} concepts...")
        self.embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"Index built successfully. Embedding shape: {self.embeddings.shape}")

    def find_candidates(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar concepts to the query text.

        Args:
            query_text: The text to search for
            top_k: Number of top candidates to return

        Returns:
            List of dictionaries with candidate information and similarity scores
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        if not query_text.strip():
            return []

        # Generate embedding for query
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True
        )[0]

        # Calculate cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build result list
        candidates = []
        for idx in top_indices:
            score = float(similarities[idx])
            row = self.df.iloc[idx]

            candidate = {
                'id': row.get('id', ''),
                'label': row.get('label', ''),
                'context': row.get('context', '')[:200] + '...' if len(row.get('context', '')) > 200 else row.get('context', ''),
                'file_path': row.get('file_path', ''),
                'score': score,
                'rank': len(candidates) + 1
            }
            candidates.append(candidate)

        return candidates

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between a vector and a matrix of vectors.

        Args:
            vec1: Single vector (1D array)
            vec2: Matrix of vectors (2D array)

        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

        # Compute dot product (cosine similarity)
        similarities = np.dot(vec2_norm, vec1_norm)

        return similarities


def main():
    """
    Test the VectorIndex with the S1000D bike data.
    """
    from data_loader import load_s1000d_data

    print("=" * 60)
    print("BERT Vector Matcher Test")
    print("=" * 60)

    # Load data
    print("\n1. Loading S1000D data...")
    df = load_s1000d_data('bike')

    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"   Loaded {len(df)} concepts")

    # Build index
    print("\n2. Building vector index...")
    index = VectorIndex(df)
    index.build_index()

    # Test queries
    print("\n3. Testing similarity search...")
    print("=" * 60)

    test_queries = [
        "How to check the brakes?",
        "Bicycle frame description",
        "Lighting system"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)

        candidates = index.find_candidates(query, top_k=3)

        for candidate in candidates:
            print(f"\nRank {candidate['rank']} | Score: {candidate['score']:.4f}")
            print(f"Label: {candidate['label']}")
            print(f"ID: {candidate['id']}")
            print(f"Context: {candidate['context']}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")


if __name__ == '__main__':
    main()
