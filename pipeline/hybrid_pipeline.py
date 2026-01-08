"""
State-of-the-Art Hybrid Ontology Matching Pipeline.

Orchestrates the 4-stage architecture:
1. Candidate Generation (KROMA + DeepOnto + String)
2. Aggregation (Weighted Voting)
3. LLM Reranking (Claude Listwise)
4. Validation (Optional)

Implements "The Best of All Worlds" approach for ontology matching.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_all_concepts
from matchers.kroma_matcher import KROMAMatcher
from matchers.deeponto_matcher import DeepOntoMatcher
from matchers.string_matcher import StringMatcher
from aggregation.weighted_aggregator import WeightedAggregator
from reranking.llm_reranker import LLMReranker


class HybridPipeline:
    """
    State-of-the-Art Hybrid Ontology Matching Pipeline.

    Combines complementary matching strategies for maximum precision and recall.
    """

    def __init__(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        use_llm: bool = True,
        aggregation_method: str = 'rank_fusion'
    ):
        """
        Initialize hybrid pipeline.

        Args:
            source_df: Source ontology concepts (S1000D)
            target_df: Target ontology concepts
            use_llm: Whether to use LLM reranking (requires API key)
            aggregation_method: 'rank_fusion' or 'weighted_sum'
        """
        self.source_df = source_df
        self.target_df = target_df
        self.use_llm = use_llm
        self.aggregation_method = aggregation_method

        print("=" * 70)
        print("INITIALIZING HYBRID PIPELINE")
        print("=" * 70)

        # Stage 1: Initialize matchers
        print("\n[STAGE 1] Initializing Matchers...")
        print("-" * 70)

        print("\n  1.1 KROMA Matcher (DMC-based heuristics)...")
        self.kroma = KROMAMatcher(source_df, target_df)

        print("\n  1.2 DeepOnto Matcher (BERT + Reasoning)...")
        self.deeponto = DeepOntoMatcher(source_df, target_df)

        print("\n  1.3 String Matcher (baseline)...")
        self.string = StringMatcher(source_df, target_df)

        # Stage 2: Initialize aggregator
        print("\n[STAGE 2] Initializing Aggregator...")
        print("-" * 70)
        self.aggregator = WeightedAggregator()

        # Stage 3: Initialize LLM reranker (if enabled)
        self.reranker = None
        if use_llm:
            print("\n[STAGE 3] Initializing LLM Reranker...")
            print("-" * 70)
            try:
                self.reranker = LLMReranker()
                print("  ✓ Claude API ready")
            except ValueError as e:
                print(f"  Warning: {e}")
                print("  → Skipping LLM reranking")
                self.use_llm = False

        print("\n" + "=" * 70)
        print("✓ PIPELINE INITIALIZED")
        print("=" * 70)

    def match_concept(
        self,
        source_concept: Dict[str, Any],
        top_k: int = 5,
        use_llm: bool = None
    ) -> Dict[str, Any]:
        """
        Match a single source concept through the full pipeline.

        Args:
            source_concept: Source concept dictionary
            top_k: Number of candidates to aggregate
            use_llm: Override instance use_llm setting

        Returns:
            Dictionary with:
            - source_uri: Source concept URI
            - selected_uri: Best match URI (or None)
            - confidence: Confidence score
            - reason: Reasoning (if LLM used)
            - aggregated_candidates: Top-k aggregated candidates
            - matcher_results: Raw matcher outputs
        """
        if use_llm is None:
            use_llm = self.use_llm

        # Stage 1: Candidate Generation (parallel matching)
        kroma_cands = self.kroma.find_candidates(source_concept, top_k=top_k)
        deeponto_cands = self.deeponto.find_candidates(source_concept, top_k=top_k)
        string_cands = self.string.find_candidates(source_concept, top_k=top_k)

        matcher_results = {
            'kroma': kroma_cands,
            'deeponto': deeponto_cands,
            'string': string_cands
        }

        # Stage 2: Aggregation
        aggregated = self.aggregator.aggregate_candidates(
            matcher_results,
            top_k=top_k,
            method=self.aggregation_method
        )

        # Prepare candidates for reranking
        aggregated_candidates = []
        for uri, score, details in aggregated:
            # Find target concept
            target_row = self.target_df[self.target_df['uri'] == uri]

            if not target_row.empty:
                target_concept = target_row.iloc[0].to_dict()
                target_concept['aggregated_score'] = score
                target_concept['matcher_details'] = details
                aggregated_candidates.append(target_concept)

        # Stage 3: LLM Reranking (if enabled)
        if use_llm and self.reranker and aggregated_candidates:
            rerank_result = self.reranker.rerank_candidates(
                source_concept,
                aggregated_candidates
            )

            if rerank_result and rerank_result['selected_uri']:
                return {
                    'source_uri': source_concept['uri'],
                    'selected_uri': rerank_result['selected_uri'],
                    'confidence': rerank_result['confidence'],
                    'reason': rerank_result['reason'],
                    'method': 'llm_reranking',
                    'aggregated_candidates': aggregated_candidates,
                    'matcher_results': matcher_results
                }
            else:
                # LLM said NULL
                return {
                    'source_uri': source_concept['uri'],
                    'selected_uri': None,
                    'confidence': rerank_result.get('confidence', 0.0) if rerank_result else 0.0,
                    'reason': rerank_result.get('reason', 'No good match') if rerank_result else 'LLM error',
                    'method': 'llm_rejected',
                    'aggregated_candidates': aggregated_candidates,
                    'matcher_results': matcher_results
                }
        else:
            # No LLM: Use top aggregated result
            if aggregated_candidates:
                top_candidate = aggregated_candidates[0]
                return {
                    'source_uri': source_concept['uri'],
                    'selected_uri': top_candidate['uri'],
                    'confidence': top_candidate['aggregated_score'],
                    'reason': f"Aggregated top match (no LLM)",
                    'method': 'aggregation_only',
                    'aggregated_candidates': aggregated_candidates,
                    'matcher_results': matcher_results
                }
            else:
                return {
                    'source_uri': source_concept['uri'],
                    'selected_uri': None,
                    'confidence': 0.0,
                    'reason': 'No candidates found',
                    'method': 'no_match',
                    'aggregated_candidates': [],
                    'matcher_results': matcher_results
                }

    def match_all(
        self,
        use_llm: bool = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Match all source concepts through the pipeline.

        Args:
            use_llm: Whether to use LLM reranking
            top_k: Number of candidates to aggregate per concept

        Returns:
            List of match results
        """
        if use_llm is None:
            use_llm = self.use_llm

        print("\n" + "=" * 70)
        print("RUNNING HYBRID PIPELINE ON ALL CONCEPTS")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Source concepts: {len(self.source_df)}")
        print(f"  Target concepts: {len(self.target_df)}")
        print(f"  Aggregation method: {self.aggregation_method}")
        print(f"  LLM reranking: {use_llm}")
        print(f"  Top-k candidates: {top_k}")

        results = []

        print(f"\nProcessing {len(self.source_df)} concepts...")
        for idx, row in tqdm(self.source_df.iterrows(), total=len(self.source_df)):
            source_concept = row.to_dict()

            result = self.match_concept(
                source_concept,
                top_k=top_k,
                use_llm=use_llm
            )

            results.append(result)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        # Summary statistics
        matches_found = sum(1 for r in results if r['selected_uri'] is not None)
        llm_used = sum(1 for r in results if r['method'] == 'llm_reranking')

        print(f"\nResults Summary:")
        print(f"  Total concepts processed: {len(results)}")
        print(f"  Matches found: {matches_found}")
        print(f"  No match: {len(results) - matches_found}")
        if use_llm:
            print(f"  LLM reranking used: {llm_used}")
            print(f"  LLM rejected: {sum(1 for r in results if r['method'] == 'llm_rejected')}")
        print(f"  Aggregation only: {sum(1 for r in results if r['method'] == 'aggregation_only')}")

        return results


def main():
    """
    Test hybrid pipeline on bike dataset.
    """
    print("=" * 70)
    print("HYBRID PIPELINE TEST")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    df = load_all_concepts('bike')

    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Ontology concepts: {len(ontology_df)}")

    # Initialize pipeline
    print("\n[2/3] Initializing pipeline...")
    pipeline = HybridPipeline(
        s1000d_df,
        ontology_df,
        use_llm=True,  # Enable LLM reranking
        aggregation_method='rank_fusion'
    )

    # Test on first 3 concepts
    print("\n[3/3] Testing on sample concepts...")
    print("=" * 70)

    for i in range(min(3, len(s1000d_df))):
        concept = s1000d_df.iloc[i].to_dict()

        print(f"\n--- Test {i+1}: {concept['label']} ---")

        result = pipeline.match_concept(concept, top_k=5, use_llm=True)

        print(f"\nResult:")
        print(f"  Method: {result['method']}")
        if result['selected_uri']:
            selected = ontology_df[ontology_df['uri'] == result['selected_uri']].iloc[0]
            print(f"  Selected: {selected['label']}")
            print(f"  Confidence: {result['confidence']:.3f}")
        else:
            print(f"  Selected: NULL (no match)")

        if result.get('reason'):
            print(f"  Reason: {result['reason'][:200]}...")

        print(f"\n  Top aggregated candidates:")
        for j, cand in enumerate(result['aggregated_candidates'][:3], 1):
            print(f"    {j}. {cand['label']} (score: {cand['aggregated_score']:.3f})")

    print("\n" + "=" * 70)
    print("Test completed!")


if __name__ == '__main__':
    main()
