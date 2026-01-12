"""
Contextual Similarity Matcher (Phase B).

Compares documents based on their full context, not just raw text.
Uses contextual entailment: "Do these concepts mean the same thing
in their respective contexts?"
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer

from .rich_document import RichDocument


class ContextualMatcher:
    """
    Matcher that compares documents in their full context.

    Instead of comparing:
      Sim(TextA, TextB)

    Compares:
      Sim(ContextA + TextA, ContextB + TextB)

    This captures whether concepts have the same meaning in their
    respective hierarchical and semantic contexts.
    """

    def __init__(
        self,
        model_name: str = 'answerdotai/ModernBERT-base',
        context_mode: str = 'full',
        cache_embeddings: bool = True
    ):
        """
        Initialize contextual matcher.

        Args:
            model_name: Embedding model to use
            context_mode: Context level ('minimal', 'medium', 'full')
            cache_embeddings: Whether to cache embeddings
        """
        self.model_name = model_name
        self.context_mode = context_mode
        self.cache_embeddings = cache_embeddings

        # Load embedding model
        print(f"  Loading contextual embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True
        )
        print(f"  ✓ Model loaded")

        # Embedding cache
        self._embedding_cache = {} if cache_embeddings else None

    def compute_contextual_similarity(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> float:
        """
        Compute contextual similarity between source and candidate.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            Similarity score (0-1)
        """
        # Get contextual embeddings
        source_embedding = self._get_contextual_embedding(source)
        candidate_embedding = self._get_contextual_embedding(candidate)

        # Compute cosine similarity
        similarity = self._cosine_similarity(source_embedding, candidate_embedding)

        return float(similarity)

    def find_top_k_contextual_matches(
        self,
        source: RichDocument,
        candidates: List[RichDocument],
        top_k: int = 10
    ) -> List[Tuple[RichDocument, float]]:
        """
        Find top-K candidates based on contextual similarity.

        Args:
            source: Source document
            candidates: List of candidate documents
            top_k: Number of top matches to return

        Returns:
            List of (candidate, similarity_score) tuples
        """
        if not candidates:
            return []

        # Compute source embedding
        source_embedding = self._get_contextual_embedding(source)

        # Compute candidate embeddings and similarities
        similarities = []
        for candidate in candidates:
            cand_embedding = self._get_contextual_embedding(candidate)
            similarity = self._cosine_similarity(source_embedding, cand_embedding)
            similarities.append((candidate, float(similarity)))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def batch_compute_similarities(
        self,
        source: RichDocument,
        candidates: List[RichDocument]
    ) -> List[float]:
        """
        Batch compute similarities for efficiency.

        Args:
            source: Source document
            candidates: List of candidates

        Returns:
            List of similarity scores
        """
        # Get source embedding
        source_embedding = self._get_contextual_embedding(source)

        # Get candidate embeddings in batch
        candidate_texts = [
            cand.get_contextual_embedding_text(self.context_mode)
            for cand in candidates
        ]

        # Batch encode
        candidate_embeddings = self.model.encode(
            candidate_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Compute similarities
        similarities = []
        for cand_emb in candidate_embeddings:
            sim = self._cosine_similarity(source_embedding, cand_emb)
            similarities.append(float(sim))

        return similarities

    def _get_contextual_embedding(self, doc: RichDocument) -> np.ndarray:
        """
        Get contextual embedding for a document.

        Args:
            doc: Rich document

        Returns:
            Embedding vector
        """
        # Check cache first
        if self._embedding_cache is not None:
            if doc.uri in self._embedding_cache:
                return self._embedding_cache[doc.uri]

        # Get contextual text
        text = doc.get_contextual_embedding_text(self.context_mode)

        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Cache embedding
        if self._embedding_cache is not None:
            self._embedding_cache[doc.uri] = embedding

        return embedding

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity
        """
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

        return float(np.dot(vec1_norm, vec2_norm))

    def clear_cache(self):
        """Clear embedding cache."""
        if self._embedding_cache is not None:
            self._embedding_cache.clear()


class HybridContextualMatcher(ContextualMatcher):
    """
    Hybrid matcher combining contextual similarity with feature-based scores.
    """

    def __init__(
        self,
        model_name: str = 'answerdotai/ModernBERT-base',
        context_mode: str = 'full',
        feature_weight: float = 0.3,
        contextual_weight: float = 0.7
    ):
        """
        Initialize hybrid contextual matcher.

        Args:
            model_name: Embedding model
            context_mode: Context level
            feature_weight: Weight for feature-based score
            contextual_weight: Weight for contextual similarity
        """
        super().__init__(model_name, context_mode)

        self.feature_weight = feature_weight
        self.contextual_weight = contextual_weight

    def compute_hybrid_similarity(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute hybrid similarity combining features and context.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            (combined_score, score_breakdown)
        """
        # Contextual similarity
        contextual_sim = self.compute_contextual_similarity(source, candidate)

        # Feature-based scores
        feature_scores = self._compute_feature_scores(source, candidate)
        feature_sim = np.mean(list(feature_scores.values()))

        # Combined score
        combined = (
            self.contextual_weight * contextual_sim +
            self.feature_weight * feature_sim
        )

        breakdown = {
            'contextual_similarity': contextual_sim,
            'feature_similarity': feature_sim,
            'combined_score': combined,
            **feature_scores
        }

        return combined, breakdown

    def _compute_feature_scores(
        self,
        source: RichDocument,
        candidate: RichDocument
    ) -> Dict[str, float]:
        """
        Compute feature-based similarity scores.

        Args:
            source: Source document
            candidate: Candidate document

        Returns:
            Dictionary of feature scores
        """
        scores = {}

        # Keyword overlap score
        if source.keywords and candidate.keywords:
            source_kw = set(kw.lower() for kw in source.keywords)
            candidate_kw = set(kw.lower() for kw in candidate.keywords)

            overlap = len(source_kw & candidate_kw)
            union = len(source_kw | candidate_kw)

            scores['keyword_jaccard'] = overlap / max(union, 1)
        else:
            scores['keyword_jaccard'] = 0.0

        # Entity overlap score
        if source.entities and candidate.entities:
            source_ent = set(e.lower() for e in source.entities)
            candidate_ent = set(e.lower() for e in candidate.entities)

            overlap = len(source_ent & candidate_ent)
            union = len(source_ent | candidate_ent)

            scores['entity_jaccard'] = overlap / max(union, 1)
        else:
            scores['entity_jaccard'] = 0.0

        # Hierarchy similarity score
        if source.hierarchy_path and candidate.hierarchy_path:
            source_hier = set(h.lower() for h in source.hierarchy_path)
            candidate_hier = set(h.lower() for h in candidate.hierarchy_path)

            overlap = len(source_hier & candidate_hier)
            union = len(source_hier | candidate_hier)

            scores['hierarchy_jaccard'] = overlap / max(union, 1)
        else:
            scores['hierarchy_jaccard'] = 0.0

        # Depth similarity (inverted distance)
        depth_diff = abs(source.depth_level - candidate.depth_level)
        scores['depth_similarity'] = 1.0 / (1.0 + depth_diff)

        # Domain match score
        if source.technical_domain and candidate.technical_domain:
            if source.technical_domain == candidate.technical_domain:
                scores['domain_match'] = 1.0
            elif source.technical_domain in candidate.technical_domain or \
                 candidate.technical_domain in source.technical_domain:
                scores['domain_match'] = 0.7
            else:
                scores['domain_match'] = 0.0
        else:
            scores['domain_match'] = 0.5  # Neutral

        return scores


def main():
    """Test contextual matcher."""
    print("=" * 70)
    print("CONTEXTUAL MATCHER TEST")
    print("=" * 70)

    # Create test documents
    source = RichDocument(
        uri="test:source:1",
        label="Front Wheel Assembly",
        raw_content="Install the front wheel assembly on the fork. Ensure proper alignment.",
        hierarchy_path=["Bicycle", "FrontAssembly", "Wheel"],
        depth_level=3,
        technical_domain="wheel_system",
        entities=["wheel", "fork", "install"],
        keywords=["wheel", "assembly", "install", "front", "fork", "alignment"],
        source="s1000d",
        parent_context="Parent: FrontAssembly - Contains fork and wheel components"
    )

    candidates = [
        RichDocument(
            uri="onto:wheel:1",
            label="Wheel",
            raw_content="A circular component that rotates on an axle...",
            hierarchy_path=["BikeComponent", "Wheel"],
            depth_level=2,
            technical_domain="wheel_system",
            entities=["wheel", "axle", "component"],
            keywords=["wheel", "circular", "axle", "rotate"],
            source="ontology"
        ),
        RichDocument(
            uri="onto:fork:1",
            label="Fork",
            raw_content="The fork holds the front wheel...",
            hierarchy_path=["BikeComponent", "FrontStructure", "Fork"],
            depth_level=3,
            technical_domain="wheel_system",
            entities=["fork", "wheel", "front"],
            keywords=["fork", "front", "wheel", "hold"],
            source="ontology"
        ),
    ]

    # Test contextual matcher
    print("\n[1/2] Testing ContextualMatcher...")
    matcher = ContextualMatcher(context_mode='full')

    print(f"\nSource: {source.label}")
    print(f"Context mode: full")

    for candidate in candidates:
        similarity = matcher.compute_contextual_similarity(source, candidate)
        print(f"\n  Candidate: {candidate.label}")
        print(f"    Contextual Similarity: {similarity:.3f}")

    # Test hybrid matcher
    print("\n[2/2] Testing HybridContextualMatcher...")
    hybrid_matcher = HybridContextualMatcher(
        context_mode='full',
        feature_weight=0.3,
        contextual_weight=0.7
    )

    for candidate in candidates:
        score, breakdown = hybrid_matcher.compute_hybrid_similarity(source, candidate)
        print(f"\n  Candidate: {candidate.label}")
        print(f"    Combined Score: {score:.3f}")
        print(f"    Breakdown:")
        for key, val in breakdown.items():
            print(f"      {key}: {val:.3f}")

    print("\n" + "=" * 70)
    print("✓ Contextual matcher test complete!")


if __name__ == '__main__':
    main()
