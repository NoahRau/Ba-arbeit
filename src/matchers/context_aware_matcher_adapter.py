"""
Context-Aware Matcher Adapter.

Wraps existing matchers (KROMA, DeepOnto, String) to work with RichDocuments
and applies context-weighted scoring.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enrichment.rich_document import RichDocument
from enrichment.blocking_filter import ContextualBlockingFilter
from enrichment.contextual_matcher import HybridContextualMatcher


class ContextAwareMatcherAdapter:
    """
    Adapter that makes existing matchers context-aware.

    Workflow:
    1. Convert RichDocuments to dict format for old matchers
    2. Apply blocking filter (Phase A)
    3. Run old matchers
    4. Apply context-weighted scoring (Phase B)
    5. Return enhanced candidates
    """

    def __init__(
        self,
        base_matcher: Any,
        use_blocking: bool = True,
        use_contextual_scoring: bool = True,
        context_weight: float = 0.4
    ):
        """
        Initialize adapter.

        Args:
            base_matcher: Original matcher (KROMA, DeepOnto, String)
            use_blocking: Apply feature-based blocking filter
            use_contextual_scoring: Apply contextual similarity scoring
            context_weight: Weight for contextual score (0-1)
        """
        self.base_matcher = base_matcher
        self.use_blocking = use_blocking
        self.use_contextual_scoring = use_contextual_scoring
        self.context_weight = context_weight

        # Initialize blocking filter
        if self.use_blocking:
            self.blocking_filter = ContextualBlockingFilter(
                max_depth_difference=2,
                min_keyword_overlap=1,
                require_domain_match=False
            )

        # Initialize contextual matcher
        if self.use_contextual_scoring:
            # Use lighter model for speed
            self.contextual_matcher = HybridContextualMatcher(
                context_mode='medium',  # Balance between full and minimal
                feature_weight=0.3,
                contextual_weight=0.7
            )

    def find_candidates(
        self,
        source: Union[RichDocument, Dict[str, Any]],
        top_k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Find candidates with context-aware enhancements.

        Args:
            source: Source RichDocument or dict
            top_k: Number of candidates to return

        Returns:
            List of (target_uri, score) tuples
        """
        # Convert to dict if RichDocument
        if isinstance(source, RichDocument):
            source_dict = self._richdoc_to_dict(source)
            source_rich = source
        else:
            source_dict = source
            source_rich = None

        # Get all target candidates from base matcher
        # Request more candidates for filtering
        initial_k = top_k * 3 if self.use_blocking else top_k

        base_candidates = self.base_matcher.find_candidates(source_dict, top_k=initial_k)

        if not base_candidates:
            return []

        # Phase A: Blocking Filter (if enabled and we have RichDocuments)
        if self.use_blocking and source_rich and hasattr(self.base_matcher, 'target_df'):
            # Get target RichDocuments for candidates
            target_rich_docs = self._get_target_richdocs(base_candidates)

            if target_rich_docs:
                # Apply blocking filter
                filtered_docs = self.blocking_filter.filter_candidates(source_rich, target_rich_docs)
                filtered_uris = set(doc.uri for doc in filtered_docs)

                # Keep only filtered candidates
                base_candidates = [(uri, score) for uri, score in base_candidates if uri in filtered_uris]

        # Phase B: Context-Weighted Scoring (if enabled)
        if self.use_contextual_scoring and source_rich and hasattr(self.base_matcher, 'target_df'):
            enhanced_candidates = self._apply_contextual_scoring(
                source_rich,
                base_candidates,
                top_k=top_k
            )
            return enhanced_candidates

        # Return top-k from base matcher
        return base_candidates[:top_k]

    def _richdoc_to_dict(self, rich_doc: RichDocument) -> Dict[str, Any]:
        """Convert RichDocument to dict format for old matchers."""
        return {
            'uri': rich_doc.uri,
            'label': rich_doc.label,
            'context_text': rich_doc.get_contextual_embedding_text('medium'),
            'source': rich_doc.source,
        }

    def _get_target_richdocs(self, candidates: List[Tuple[str, float]]) -> List[RichDocument]:
        """
        Get RichDocuments for target candidates.

        Args:
            candidates: List of (uri, score) tuples

        Returns:
            List of RichDocuments (if available from matcher)
        """
        if not hasattr(self.base_matcher, 'target_df'):
            return []

        target_df = self.base_matcher.target_df
        candidate_uris = [uri for uri, _ in candidates]

        # Check if target_df has RichDocument data
        target_rich_docs = []

        for uri in candidate_uris:
            matching_rows = target_df[target_df['uri'] == uri]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]

                # If row is already a RichDocument-like object
                if hasattr(row, 'to_dict') and hasattr(row, 'hierarchy_path'):
                    target_rich_docs.append(row)
                # Otherwise create basic RichDocument
                else:
                    from enrichment.feature_extractor import FeatureExtractor
                    extractor = FeatureExtractor()
                    rich_doc = extractor.enrich_from_dataframe_row(row, source='target')
                    target_rich_docs.append(rich_doc)

        return target_rich_docs

    def _apply_contextual_scoring(
        self,
        source: RichDocument,
        base_candidates: List[Tuple[str, float]],
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Apply contextual similarity scoring on top of base scores.

        Args:
            source: Source RichDocument
            base_candidates: Candidates from base matcher with scores
            top_k: Number to return

        Returns:
            List of (uri, combined_score) tuples
        """
        if not base_candidates:
            return []

        # Get RichDocuments for candidates
        target_docs = self._get_target_richdocs(base_candidates)

        if not target_docs:
            # Fallback to base scores
            return base_candidates[:top_k]

        # Create mapping: uri -> base_score
        base_score_map = {uri: score for uri, score in base_candidates}

        # Compute hybrid scores (contextual + features)
        enhanced_candidates = []

        for target_doc in target_docs:
            # Get base score
            base_score = base_score_map.get(target_doc.uri, 0.0)

            # Compute contextual score
            contextual_score, breakdown = self.contextual_matcher.compute_hybrid_similarity(
                source,
                target_doc
            )

            # Combine scores
            combined_score = (
                (1 - self.context_weight) * base_score +
                self.context_weight * contextual_score
            )

            enhanced_candidates.append((target_doc.uri, combined_score))

        # Sort by combined score
        enhanced_candidates.sort(key=lambda x: x[1], reverse=True)

        return enhanced_candidates[:top_k]


def wrap_matcher_with_context_awareness(
    matcher: Any,
    use_blocking: bool = True,
    use_contextual_scoring: bool = True
) -> ContextAwareMatcherAdapter:
    """
    Helper to wrap an existing matcher with context-awareness.

    Args:
        matcher: Matcher to wrap (KROMA, DeepOnto, String)
        use_blocking: Enable blocking filter
        use_contextual_scoring: Enable contextual scoring

    Returns:
        Context-aware wrapped matcher
    """
    return ContextAwareMatcherAdapter(
        matcher,
        use_blocking=use_blocking,
        use_contextual_scoring=use_contextual_scoring,
        context_weight=0.4  # 40% contextual, 60% base matcher
    )


# Import Union for type hints
from typing import Union


def main():
    """Test context-aware matcher adapter."""
    print("=" * 70)
    print("CONTEXT-AWARE MATCHER ADAPTER TEST")
    print("=" * 70)

    print("\nThis adapter wraps existing matchers with context-awareness:")
    print("  1. Feature-based blocking (Phase A)")
    print("  2. Contextual similarity scoring (Phase B)")
    print("  3. Context-weighted score combination")

    print("\nTo use with existing pipeline:")
    print("  from matchers.context_aware_matcher_adapter import wrap_matcher_with_context_awareness")
    print("  ")
    print("  # Wrap existing matcher")
    print("  deeponto_matcher = DeepOntoMatcher(source_df, target_df)")
    print("  context_aware_deeponto = wrap_matcher_with_context_awareness(deeponto_matcher)")
    print("  ")
    print("  # Use as normal")
    print("  candidates = context_aware_deeponto.find_candidates(source_concept, top_k=60)")

    print("\n" + "=" * 70)
    print("✓ Context-aware adapter ready!")


if __name__ == '__main__':
    main()
