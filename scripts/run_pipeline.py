#!/usr/bin/env python3
"""
Run the full hybrid pipeline on all S1000D concepts.

This script:
1. Loads S1000D and BikeOntology data
2. Runs the hybrid pipeline (KROMA + DeepOnto + String → Aggregation → LLM)
3. Saves results to CSV file
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_all_concepts
from src.pipeline.hybrid_pipeline import HybridPipeline
from config import PIPELINE_CONFIG, RESULTS_DIR

def main():
    """Run the full pipeline and save results."""

    print("=" * 80)
    print("HYBRID ONTOLOGY MATCHING PIPELINE - FULL RUN")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_all_concepts(include_ontologies=True)

    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Ontology concepts: {len(ontology_df)}")

    # Initialize pipeline
    print("\n[2/4] Initializing pipeline...")
    print(f"  Matcher top-k: {PIPELINE_CONFIG['matcher_top_k']}")
    print(f"  Aggregation top-k: {PIPELINE_CONFIG['aggregation_top_k']}")
    print(f"  Neural reranker top-k: {PIPELINE_CONFIG['reranker_top_k']}")
    print(f"  Neural reranker enabled: {PIPELINE_CONFIG['use_neural_reranker']}")
    print(f"  LLM enabled: {PIPELINE_CONFIG['use_llm']}")

    pipeline = HybridPipeline(
        s1000d_df,
        ontology_df,
        use_neural_reranker=PIPELINE_CONFIG['use_neural_reranker'],
        use_llm=PIPELINE_CONFIG['use_llm'],
        aggregation_method=PIPELINE_CONFIG['aggregation_method'],
        matcher_top_k=PIPELINE_CONFIG['matcher_top_k'],
        aggregation_top_k=PIPELINE_CONFIG['aggregation_top_k'],
        reranker_top_k=PIPELINE_CONFIG['reranker_top_k']
    )

    # Run pipeline on all concepts
    print("\n[3/4] Running pipeline on all concepts...")
    results = pipeline.match_all(use_llm=PIPELINE_CONFIG['use_llm'])

    # Convert to DataFrame
    print("\n[4/4] Saving results...")

    results_data = []
    for result in results:
        # Get source info
        source_row = s1000d_df[s1000d_df['uri'] == result['source_uri']].iloc[0]

        # Get target info (if matched)
        target_label = None
        if result['selected_uri']:
            target_rows = ontology_df[ontology_df['uri'] == result['selected_uri']]
            if not target_rows.empty:
                target_label = target_rows.iloc[0]['label']

        results_data.append({
            'source_uri': result['source_uri'],
            'source_label': source_row['label'],
            'selected_uri': result['selected_uri'],
            'selected_label': target_label,
            'confidence': result['confidence'],
            'method': result['method'],
            'reason': result.get('reason', ''),
        })

    results_df = pd.DataFrame(results_data)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f'pipeline_matches_v4_neural_{timestamp}.csv'

    results_df.to_csv(output_file, index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE RUN COMPLETE")
    print("=" * 80)

    total = len(results_df)
    matches = results_df['selected_uri'].notna().sum()
    no_matches = total - matches

    print(f"\nTotal concepts: {total}")
    print(f"  Matches found: {matches} ({100*matches/total:.1f}%)")
    print(f"  No matches: {no_matches} ({100*no_matches/total:.1f}%)")

    if matches > 0:
        avg_confidence = results_df[results_df['selected_uri'].notna()]['confidence'].mean()
        print(f"  Average confidence: {avg_confidence:.3f}")

    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
