"""
Weighted Aggregator for Multi-Matcher Fusion.

Combines scores from multiple matchers using weighted voting:
- KROMA: Weight 0.40 (DMC code reliability)
- DeepOnto: Weight 0.35 (semantic understanding)
- StringMatcher: Weight 0.25 (string baseline)

Implements rank-based fusion and score normalization.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd


class WeightedAggregator:
    """
    Aggregates matcher outputs via weighted voting.

    Supports:
    - Score-based aggregation (weighted sum)
    - Rank-based aggregation (for robustness)
    - Score normalization
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize aggregator with matcher weights.

        Args:
            weights: Dictionary mapping matcher names to weights
                Default: {'kroma': 0.40, 'deeponto': 0.35, 'string': 0.25}
        """
        self.weights = weights or {
            'kroma': 0.40,      # High weight for DMC-based matching
            'deeponto': 0.35,   # Semantic understanding
            'string': 0.25      # String baseline
        }

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        print(f"  Aggregator weights: {self.weights}")

    def aggregate_candidates(
        self,
        matcher_results: Dict[str, List[Tuple[str, float]]],
        top_k: int = 5,
        method: str = 'weighted_sum'
    ) -> List[Tuple[str, float, Dict]]:
        """
        Aggregate candidates from multiple matchers.

        Args:
            matcher_results: Dictionary mapping matcher_name to list of (uri, score) tuples
                Example: {
                    'kroma': [('uri1', 0.8), ('uri2', 0.6), ...],
                    'deeponto': [('uri1', 0.9), ('uri3', 0.7), ...],
                    'string': [('uri2', 0.5), ...]
                }
            top_k: Number of final candidates to return
            method: Aggregation method ('weighted_sum' or 'rank_fusion')

        Returns:
            List of (uri, aggregated_score, details_dict) tuples
            - uri: Target concept URI
            - aggregated_score: Final combined score
            - details_dict: Individual matcher scores and ranks
        """
        if method == 'weighted_sum':
            return self._aggregate_weighted_sum(matcher_results, top_k)
        elif method == 'rank_fusion':
            return self._aggregate_rank_fusion(matcher_results, top_k)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _aggregate_weighted_sum(
        self,
        matcher_results: Dict[str, List[Tuple[str, float]]],
        top_k: int
    ) -> List[Tuple[str, float, Dict]]:
        """
        Weighted sum aggregation.

        Combines normalized scores with matcher weights.
        """
        # Collect all unique URIs
        all_uris = set()
        for matcher_name, candidates in matcher_results.items():
            for uri, score in candidates:
                all_uris.add(uri)

        # Compute aggregated scores
        aggregated = []

        for uri in all_uris:
            # Get scores from each matcher
            scores_dict = {}
            raw_scores = {}

            for matcher_name, candidates in matcher_results.items():
                # Find score for this URI
                score = self._get_score(uri, candidates)
                raw_scores[matcher_name] = score

                # Normalize score within matcher's range
                # (Some matchers have different score distributions)
                normalized_score = self._normalize_score(score, candidates)
                scores_dict[matcher_name] = normalized_score

            # Compute weighted sum
            final_score = sum(
                scores_dict.get(matcher_name, 0.0) * weight
                for matcher_name, weight in self.weights.items()
            )

            # Store details
            details = {
                'raw_scores': raw_scores,
                'normalized_scores': scores_dict,
                'weights': self.weights.copy()
            }

            aggregated.append((uri, final_score, details))

        # Sort by final score descending
        aggregated.sort(key=lambda x: x[1], reverse=True)

        return aggregated[:top_k]

    def _aggregate_rank_fusion(
        self,
        matcher_results: Dict[str, List[Tuple[str, float]]],
        top_k: int
    ) -> List[Tuple[str, float, Dict]]:
        """
        Rank-based fusion (Reciprocal Rank Fusion).

        More robust to score scale differences.
        RRF formula: score = sum(weight / (k + rank))
        where k=60 is a constant.
        """
        k = 60  # RRF constant

        # Collect all URIs
        all_uris = set()
        for matcher_name, candidates in matcher_results.items():
            for uri, score in candidates:
                all_uris.add(uri)

        # Compute RRF scores
        aggregated = []

        for uri in all_uris:
            rrf_score = 0.0
            ranks_dict = {}

            for matcher_name, candidates in matcher_results.items():
                # Find rank of this URI (1-indexed)
                rank = self._get_rank(uri, candidates)

                if rank is not None:
                    # RRF formula with weight
                    weight = self.weights.get(matcher_name, 0.0)
                    rrf_contribution = weight / (k + rank)
                    rrf_score += rrf_contribution
                    ranks_dict[matcher_name] = rank
                else:
                    ranks_dict[matcher_name] = None

            details = {
                'ranks': ranks_dict,
                'rrf_score': rrf_score,
                'weights': self.weights.copy()
            }

            aggregated.append((uri, rrf_score, details))

        # Sort by RRF score descending
        aggregated.sort(key=lambda x: x[1], reverse=True)

        return aggregated[:top_k]

    def _get_score(self, uri: str, candidates: List[Tuple[str, float]]) -> float:
        """Get score for URI from candidate list, return 0.0 if not found."""
        for cand_uri, score in candidates:
            if cand_uri == uri:
                return score
        return 0.0

    def _get_rank(self, uri: str, candidates: List[Tuple[str, float]]) -> int:
        """Get rank of URI in candidate list (1-indexed), None if not found."""
        for rank, (cand_uri, score) in enumerate(candidates, start=1):
            if cand_uri == uri:
                return rank
        return None

    def _normalize_score(
        self,
        score: float,
        candidates: List[Tuple[str, float]]
    ) -> float:
        """
        Normalize score to [0, 1] based on candidate list distribution.

        Uses min-max normalization within the matcher's score range.
        """
        if not candidates:
            return 0.0

        scores = [s for uri, s in candidates]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return 1.0 if score > 0 else 0.0

        # Min-max normalization
        normalized = (score - min_score) / (max_score - min_score)

        return normalized


def main():
    """
    Test Weighted Aggregator with mock matcher results.
    """
    print("=" * 70)
    print("WEIGHTED AGGREGATOR TEST")
    print("=" * 70)

    # Mock matcher results
    mock_results = {
        'kroma': [
            ('uri:Wheel', 0.42),
            ('uri:Hub', 0.38),
            ('uri:Rim', 0.35),
        ],
        'deeponto': [
            ('uri:Wheel', 0.85),
            ('uri:Tire', 0.82),
            ('uri:Frame', 0.78),
        ],
        'string': [
            ('uri:Wheel', 0.25),
            ('uri:WheelAssembly', 0.22),
            ('uri:Hub', 0.18),
        ]
    }

    print("\nInput matcher results:")
    for matcher_name, candidates in mock_results.items():
        print(f"\n  {matcher_name}:")
        for uri, score in candidates:
            print(f"    - {uri}: {score:.3f}")

    # Test weighted sum
    print("\n" + "=" * 70)
    print("METHOD 1: Weighted Sum Aggregation")
    print("=" * 70)

    aggregator = WeightedAggregator()
    results_weighted = aggregator.aggregate_candidates(
        mock_results,
        top_k=5,
        method='weighted_sum'
    )

    print("\nAggregated results:")
    for i, (uri, score, details) in enumerate(results_weighted, 1):
        print(f"\n{i}. {uri}")
        print(f"   Final Score: {score:.3f}")
        print(f"   Raw Scores: {details['raw_scores']}")
        print(f"   Normalized: {details['normalized_scores']}")

    # Test rank fusion
    print("\n" + "=" * 70)
    print("METHOD 2: Rank Fusion Aggregation")
    print("=" * 70)

    results_rank = aggregator.aggregate_candidates(
        mock_results,
        top_k=5,
        method='rank_fusion'
    )

    print("\nAggregated results:")
    for i, (uri, score, details) in enumerate(results_rank, 1):
        print(f"\n{i}. {uri}")
        print(f"   RRF Score: {score:.3f}")
        print(f"   Ranks: {details['ranks']}")

    print("\n" + "=" * 70)
    print("Test completed!")


if __name__ == '__main__':
    main()
