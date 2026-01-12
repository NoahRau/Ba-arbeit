"""
Neural Reranker using BGE Reranker v2 (BAAI).

This reranker uses a cross-encoder model to rescore candidates
based on the full context of source and target concepts.

Model: BAAI/bge-reranker-v2-m3
- Cross-encoder architecture (compares query + document together)
- More accurate than bi-encoder similarity
- Slower but better for final candidate filtering
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from sentence_transformers import CrossEncoder

try:
    from ..enrichment.rich_document import RichDocument
except ImportError:
    RichDocument = None  # Fallback if enrichment module not available


class NeuralReranker:
    """
    Neural reranker using BGE cross-encoder for accurate candidate scoring.

    Usage:
        reranker = NeuralReranker()
        top_candidates = reranker.rerank(source_concept, candidates, top_k=7)
    """

    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-v2-m3',
        device: str = None,
        use_contextual_mode: bool = False
    ):
        """
        Initialize neural reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            use_contextual_mode: If True, use [Context A + Text A] <SEP> [Context B + Text B] format
        """
        self.model_name = model_name
        self.use_contextual_mode = use_contextual_mode

        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"  Loading {model_name}...")
        print(f"  Device: {self.device}")
        print(f"  Contextual mode: {use_contextual_mode}")

        # Load cross-encoder model
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            device=self.device
        )

        print(f"  ✓ Neural reranker loaded")

    def rerank(
        self,
        source_concept: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_k: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder scoring.

        Args:
            source_concept: Source concept dict with 'label', 'context_text'
            candidates: List of candidate dicts with 'label', 'context_text', 'aggregated_score'
            top_k: Number of top candidates to return

        Returns:
            Top-k candidates with updated 'reranker_score'
        """
        if not candidates:
            return []

        # Create query text from source concept
        query = self._create_query_text(source_concept)

        # Create document texts from candidates
        documents = [self._create_document_text(cand) for cand in candidates]

        # Create query-document pairs for cross-encoder
        pairs = [(query, doc) for doc in documents]

        # Get reranker scores (higher = better match)
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        # Normalize scores to [0, 1] range using sigmoid
        scores = self._normalize_scores(scores)

        # Add reranker scores to candidates
        reranked_candidates = []
        for i, candidate in enumerate(candidates):
            candidate_copy = candidate.copy()
            candidate_copy['reranker_score'] = float(scores[i])

            # Combine with aggregated score (70% reranker, 30% aggregated)
            combined_score = 0.7 * scores[i] + 0.3 * candidate.get('aggregated_score', 0.0)
            candidate_copy['combined_score'] = float(combined_score)

            reranked_candidates.append(candidate_copy)

        # Sort by combined score
        reranked_candidates.sort(key=lambda x: x['combined_score'], reverse=True)

        # Return top-k
        return reranked_candidates[:top_k]

    def rerank_contextual(
        self,
        source_doc: Union['RichDocument', Dict[str, Any]],
        candidate_docs: List[Union['RichDocument', Dict[str, Any]]],
        top_k: int = 7
    ) -> List[Union['RichDocument', Dict[str, Any]]]:
        """
        Rerank candidates using CONTEXTUAL cross-encoder scoring.

        Uses format: [Context A + Candidate A] <SEP> [Context B + Candidate B]

        Args:
            source_doc: Source RichDocument or dict
            candidate_docs: List of candidate RichDocuments or dicts
            top_k: Number of top candidates to return

        Returns:
            Top-k candidates with updated scores
        """
        if not candidate_docs:
            return []

        # Check if source is RichDocument by checking for attributes
        source_is_rich = hasattr(source_doc, 'hierarchy_path') and hasattr(source_doc, 'raw_content')

        # Create contextual query-document pairs
        pairs = []
        for candidate in candidate_docs:
            # Check if candidate is RichDocument
            candidate_is_rich = hasattr(candidate, 'hierarchy_path') and hasattr(candidate, 'raw_content')

            if source_is_rich and candidate_is_rich:
                # Use full context from RichDocument
                query_text = self._create_contextual_text(source_doc)
                doc_text = self._create_contextual_text(candidate)
            else:
                # Fallback to dict-based
                query_text = self._create_query_text(source_doc)
                doc_text = self._create_document_text(candidate)

            pairs.append((query_text, doc_text))

        # Get reranker scores
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        # Normalize scores
        scores = self._normalize_scores(scores)

        # Add scores to candidates
        reranked = []
        for i, candidate in enumerate(candidate_docs):
            # Check if it's a dict or RichDocument
            if hasattr(candidate, 'to_dict'):
                # RichDocument - convert to dict
                candidate_copy = candidate.to_dict()
            elif isinstance(candidate, dict):
                candidate_copy = candidate.copy()
            else:
                # Unknown type - create minimal dict
                candidate_copy = {'uri': str(candidate), 'label': str(candidate)}

            candidate_copy['reranker_score'] = float(scores[i])

            # Combine with existing score
            existing_score = candidate_copy.get('aggregated_score', 0.0)
            combined = 0.7 * scores[i] + 0.3 * existing_score
            candidate_copy['combined_score'] = float(combined)

            reranked.append(candidate_copy)

        # Sort and return top-k
        reranked.sort(key=lambda x: x['combined_score'], reverse=True)

        return reranked[:top_k]

    def _create_contextual_text(self, doc: 'RichDocument') -> str:
        """
        Create contextual text from RichDocument.

        Format: [CONTEXT] hierarchy + parent + ... [TEXT] label + content

        Args:
            doc: RichDocument

        Returns:
            Contextual text string
        """
        parts = []

        # Add hierarchy context
        if doc.hierarchy_path:
            hierarchy = ' > '.join(doc.hierarchy_path)
            parts.append(f"[HIERARCHY] {hierarchy}")

        # Add parent context
        if doc.parent_context:
            parts.append(f"[PARENT] {doc.parent_context}")

        # Add domain
        if doc.technical_domain:
            parts.append(f"[DOMAIN] {doc.technical_domain}")

        # Add main content
        parts.append(f"[CONTENT] {doc.label}: {doc.raw_content[:200]}")

        # Add entities
        if doc.entities:
            entities = ', '.join(doc.entities[:5])
            parts.append(f"[ENTITIES] {entities}")

        return ' | '.join(parts)[:500]  # Limit to 500 chars

    def _create_query_text(self, concept: Dict[str, Any]) -> str:
        """
        Create query text from source concept.

        Args:
            concept: Source concept dict

        Returns:
            Query text string
        """
        label = concept.get('label', '')
        context = concept.get('context_text', '')

        # Use hierarchical context if available
        if context and len(context) > len(label):
            # Limit to 300 chars for efficiency
            return context[:300] if len(context) > 300 else context
        else:
            return label

    def _create_document_text(self, candidate: Dict[str, Any]) -> str:
        """
        Create document text from candidate.

        Args:
            candidate: Candidate dict

        Returns:
            Document text string
        """
        label = candidate.get('label', '')
        context = candidate.get('context_text', '')

        # Use hierarchical context if available
        if context and len(context) > len(label):
            # Limit to 300 chars for efficiency
            return context[:300] if len(context) > 300 else context
        else:
            return label

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using sigmoid.

        Args:
            scores: Raw cross-encoder scores

        Returns:
            Normalized scores
        """
        # Apply sigmoid to map to [0, 1]
        return 1.0 / (1.0 + np.exp(-scores))


def main():
    """Test neural reranker."""
    print("=" * 70)
    print("NEURAL RERANKER TEST")
    print("=" * 70)

    # Initialize reranker
    print("\n[1/2] Initializing neural reranker...")
    reranker = NeuralReranker()

    # Test data
    print("\n[2/2] Testing reranking...")

    source = {
        'uri': 'test:wheel',
        'label': 'Wheel - Description of how it is made',
        'context_text': 'BikeComponent > Wheel'
    }

    candidates = [
        {
            'uri': 'onto:wheel',
            'label': 'Wheel',
            'context_text': 'BikeComponent > Wheel',
            'aggregated_score': 0.85
        },
        {
            'uri': 'onto:tire',
            'label': 'Tire',
            'context_text': 'BikeComponent > Wheel > Tire',
            'aggregated_score': 0.75
        },
        {
            'uri': 'onto:frame',
            'label': 'Frame',
            'context_text': 'BikeComponent > Frame',
            'aggregated_score': 0.70
        },
    ]

    print(f"\nSource: {source['label']}")
    print(f"\nCandidates ({len(candidates)}):")
    for i, cand in enumerate(candidates, 1):
        print(f"  {i}. {cand['label']:20s} (aggregated: {cand['aggregated_score']:.3f})")

    # Rerank
    reranked = reranker.rerank(source, candidates, top_k=2)

    print(f"\nReranked Top-2:")
    for i, cand in enumerate(reranked, 1):
        print(f"  {i}. {cand['label']:20s} "
              f"(reranker: {cand['reranker_score']:.3f}, "
              f"combined: {cand['combined_score']:.3f})")

    print("\n" + "=" * 70)
    print("✓ Neural reranker test complete!")


if __name__ == '__main__':
    main()
