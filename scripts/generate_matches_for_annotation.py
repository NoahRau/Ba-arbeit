"""
Generate Matches for Manual Annotation.

Runs hybrid pipeline on all S1000D concepts and generates
matches for manual validation and evaluation.

Output: CSV file with matches for manual annotation
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_all_concepts
from src.pipeline.hybrid_pipeline import HybridPipeline


def generate_matches_for_annotation(
    use_llm: bool = True,
    output_file: str = None,
    top_k_candidates: int = 5
):
    """
    Generate matches for manual annotation.

    Args:
        use_llm: Whether to use LLM reranking
        output_file: Output CSV file path
        top_k_candidates: Number of candidates to show per concept

    Returns:
        DataFrame with matches
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"matches_for_annotation_{timestamp}.csv"

    print("=" * 70)
    print("HYBRID PIPELINE - MATCH GENERATION FOR ANNOTATION")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_all_concepts()  # Uses default path: data/s1000d

    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Ontology concepts: {len(ontology_df)}")

    # Initialize pipeline
    print("\n[2/4] Initializing Hybrid Pipeline...")
    pipeline = HybridPipeline(
        s1000d_df,
        ontology_df,
        use_llm=use_llm,
        aggregation_method='rank_fusion'
    )

    # Generate matches
    print(f"\n[3/4] Generating matches (LLM: {use_llm})...")
    results = pipeline.match_all(use_llm=use_llm, top_k=top_k_candidates)

    # Prepare annotation data
    print("\n[4/4] Preparing annotation file...")
    annotation_data = []

    for result in results:
        source_uri = result['source_uri']
        selected_uri = result.get('selected_uri')

        # Get source concept
        source_row = s1000d_df[s1000d_df['uri'] == source_uri]
        if source_row.empty:
            continue

        source_concept = source_row.iloc[0]

        # Prepare row for annotation
        row = {
            # Source info
            's1000d_uri': source_uri,
            's1000d_label': source_concept['label'],
            's1000d_context': source_concept.get('context_text', '')[:200],

            # Pipeline result
            'pipeline_selected_uri': selected_uri if selected_uri else 'NULL',
            'pipeline_confidence': result.get('confidence', 0.0),
            'pipeline_method': result.get('method', ''),

            # Selected match details (if exists)
            'selected_label': '',
            'selected_context': '',

            # Top-k candidates
            'candidate_1_uri': '',
            'candidate_1_label': '',
            'candidate_1_score': '',
            'candidate_2_uri': '',
            'candidate_2_label': '',
            'candidate_2_score': '',
            'candidate_3_uri': '',
            'candidate_3_label': '',
            'candidate_3_score': '',
            'candidate_4_uri': '',
            'candidate_4_label': '',
            'candidate_4_score': '',
            'candidate_5_uri': '',
            'candidate_5_label': '',
            'candidate_5_score': '',

            # For manual annotation
            'is_match_manual': '',  # User fills: TRUE/FALSE
            'correct_match_uri': '',  # If FALSE, user can provide correct URI
            'notes': ''  # Optional notes
        }

        # Fill selected match details
        if selected_uri:
            selected_row = ontology_df[ontology_df['uri'] == selected_uri]
            if not selected_row.empty:
                selected_concept = selected_row.iloc[0]
                row['selected_label'] = selected_concept['label']
                row['selected_context'] = selected_concept.get('context_text', '')[:200]

        # Fill top-k candidates
        candidates = result.get('aggregated_candidates', [])
        for i, candidate in enumerate(candidates[:5], 1):
            row[f'candidate_{i}_uri'] = candidate['uri']
            row[f'candidate_{i}_label'] = candidate['label']
            row[f'candidate_{i}_score'] = f"{candidate.get('aggregated_score', 0.0):.3f}"

        annotation_data.append(row)

    # Create DataFrame
    annotation_df = pd.DataFrame(annotation_data)

    # Save to CSV
    annotation_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✓ Generated {len(annotation_df)} matches")
    print(f"✓ Saved to: {output_file}")

    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    matches_found = (annotation_df['pipeline_selected_uri'] != 'NULL').sum()
    no_match = (annotation_df['pipeline_selected_uri'] == 'NULL').sum()

    print(f"\nPipeline Results:")
    print(f"  Total concepts: {len(annotation_df)}")
    print(f"  Matches found: {matches_found} ({matches_found/len(annotation_df)*100:.1f}%)")
    print(f"  No match (NULL): {no_match} ({no_match/len(annotation_df)*100:.1f}%)")

    if use_llm:
        llm_used = (annotation_df['pipeline_method'] == 'llm_reranking').sum()
        llm_rejected = (annotation_df['pipeline_method'] == 'llm_rejected').sum()
        print(f"\nLLM Usage:")
        print(f"  LLM accepted: {llm_used}")
        print(f"  LLM rejected: {llm_rejected}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"\n1. Open {output_file} in Excel/Google Sheets")
    print(f"2. For each row, fill 'is_match_manual' column:")
    print(f"   - TRUE if pipeline_selected_uri is correct")
    print(f"   - FALSE if wrong or NULL when should be a match")
    print(f"3. If FALSE, optionally fill 'correct_match_uri' from candidates")
    print(f"4. Save the annotated file")
    print(f"5. Run evaluation script on the annotated file")

    return annotation_df


def main():
    """Run match generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate matches for manual annotation')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM reranking')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--top-k', type=int, default=5, help='Number of candidates per concept')

    args = parser.parse_args()

    generate_matches_for_annotation(
        use_llm=not args.no_llm,
        output_file=args.output,
        top_k_candidates=args.top_k
    )


if __name__ == '__main__':
    main()
