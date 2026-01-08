"""
Simple String-Based Matcher (AML Replacement).

Fast baseline matcher using string similarity metrics:
- Exact match
- Token overlap (Jaccard)
- Edit distance (Levenshtein)
- Substring matching

Serves as lightweight alternative when AML is unavailable.
"""

import re
from typing import List, Tuple, Dict, Any
import pandas as pd
from difflib import SequenceMatcher

try:
    from .base_matcher import BaseMatcher
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from matchers.base_matcher import BaseMatcher


class StringMatcher(BaseMatcher):
    """
    Lightweight string-based matcher.

    Uses multiple string similarity metrics as AML alternative.
    """

    def __init__(self, source_df: pd.DataFrame, target_df: pd.DataFrame):
        super().__init__(source_df, target_df)

        # Precompute normalized labels for faster matching
        self.target_labels_normalized = []
        for idx, row in target_df.iterrows():
            label = self._normalize_string(row.get('label', ''))
            self.target_labels_normalized.append(label)

    def _normalize_string(self, text: str) -> str:
        """Normalize string for comparison."""
        # Lowercase
        text = text.lower()
        # Remove special chars
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

    def _tokenize(self, text: str) -> set:
        """Tokenize and normalize text."""
        return set(self._normalize_string(text).split())

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _sequence_similarity(self, str1: str, str2: str) -> float:
        """Compute sequence similarity (similar to edit distance)."""
        return SequenceMatcher(None, str1, str2).ratio()

    def _compute_string_score(
        self,
        source_label: str,
        target_label: str,
        source_context: str = '',
        target_context: str = ''
    ) -> float:
        """
        Compute overall string similarity score.

        Combines multiple metrics with weights.
        """
        # Normalize
        src_norm = self._normalize_string(source_label)
        tgt_norm = self._normalize_string(target_label)

        # 1. Exact match (after normalization)
        if src_norm == tgt_norm:
            return 1.0

        # 2. One contains the other (substring)
        if src_norm in tgt_norm or tgt_norm in src_norm:
            return 0.85

        # 3. Token overlap (Jaccard)
        src_tokens = self._tokenize(source_label)
        tgt_tokens = self._tokenize(target_label)
        jaccard = self._jaccard_similarity(src_tokens, tgt_tokens)

        # 4. Sequence similarity
        seq_sim = self._sequence_similarity(src_norm, tgt_norm)

        # 5. Context overlap (if available)
        context_sim = 0.0
        if source_context and target_context:
            src_ctx_tokens = self._tokenize(source_context)
            tgt_ctx_tokens = self._tokenize(target_context)
            context_sim = self._jaccard_similarity(src_ctx_tokens, tgt_ctx_tokens)

        # Weighted combination
        score = (
            0.40 * jaccard +        # Token overlap is important
            0.35 * seq_sim +         # Character-level similarity
            0.25 * context_sim       # Context helps
        )

        return score

    def find_candidates(
        self,
        source_concept: Dict[str, Any],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find top-k candidates using string similarity.

        Args:
            source_concept: Source concept dictionary
            top_k: Number of candidates to return

        Returns:
            List of (target_uri, score) tuples
        """
        source_label = source_concept.get('label', '')
        source_context = source_concept.get('context_text', '')

        if not source_label:
            return []

        # Compute scores for all targets
        candidates = []
        for idx, target_row in self.target_df.iterrows():
            target_label = target_row.get('label', '')
            target_context = target_row.get('context_text', '')

            score = self._compute_string_score(
                source_label,
                target_label,
                source_context,
                target_context
            )

            if score > 0.1:  # Only keep non-trivial scores
                candidates.append((target_row['uri'], score))

        # Sort and return top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def batch_match(
        self,
        source_concepts: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch matching."""
        results = {}
        for concept in source_concepts:
            results[concept['uri']] = self.find_candidates(concept, top_k)
        return results


def main():
    """Test String Matcher."""
    from data_loader import load_all_concepts

    print("=" * 70)
    print("STRING MATCHER TEST (AML Alternative)")
    print("=" * 70)

    # Load data
    print("\n[1/2] Loading data...")
    df = load_all_concepts('bike')

    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Ontology concepts: {len(ontology_df)}")

    # Initialize
    print("\n[2/2] Initializing String Matcher...")
    matcher = StringMatcher(s1000d_df, ontology_df)
    print("  ✓ Ready")

    # Test
    print("\n" + "=" * 70)
    print("Testing string matching...")

    for i in range(min(3, len(s1000d_df))):
        concept = s1000d_df.iloc[i].to_dict()
        print(f"\n--- {concept['label']} ---")

        candidates = matcher.find_candidates(concept, top_k=5)

        for j, (uri, score) in enumerate(candidates, 1):
            target = ontology_df[ontology_df['uri'] == uri].iloc[0]
            print(f"{j}. Score: {score:.3f} - {target['label']}")

    print("\n" + "=" * 70)
    print("Test completed!")


if __name__ == '__main__':
    main()
