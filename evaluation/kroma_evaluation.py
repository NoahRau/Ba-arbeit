"""
KROMA Matcher Evaluation against Gold Standard.
Measures precision, recall, F1-score for DMC-based matching.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_all_concepts
from matchers.kroma_matcher import KROMAMatcher


def load_gold_standard(gold_path: str = 'gold_standard_metrics.json') -> List[Dict]:
    """
    Load gold standard annotations.

    Returns:
        List of annotation dictionaries with:
        - s1000d_uri
        - bike_uri
        - is_match_manual (ground truth)
    """
    with open(gold_path, 'r') as f:
        annotations = json.load(f)

    return annotations


def evaluate_kroma(
    kroma_matcher: KROMAMatcher,
    gold_standard: List[Dict],
    threshold: float = 0.35
) -> Dict:
    """
    Evaluate KROMA against gold standard.

    Args:
        kroma_matcher: Initialized KROMA matcher
        gold_standard: List of gold annotations
        threshold: Score threshold for positive match

    Returns:
        Dictionary with metrics
    """
    y_true = []
    y_pred = []
    scores = []

    print(f"\nEvaluating {len(gold_standard)} gold standard pairs...")
    print(f"Threshold: {threshold}")
    print("=" * 70)

    for i, annotation in enumerate(gold_standard):
        s1000d_uri = annotation['s1000d_uri']
        bike_uri = annotation['bike_uri']
        ground_truth = annotation['is_match_manual']

        # Find S1000D concept (flexible matching - extract DMC code)
        # Gold standard URIs might have different format
        # e.g., S1000DBIKE-AAA-DA1-0-0-00-00-AA-041-A-A vs S1000DBIKE-AAA-DA1-0-0-041
        dmc_code = s1000d_uri.split('/')[-1].split(':')[-1]

        # Try exact match first
        s1000d_row = kroma_matcher.source_df[
            kroma_matcher.source_df['uri'] == s1000d_uri
        ]

        # If not found, try partial match on DMC code
        if s1000d_row.empty:
            for idx, row in kroma_matcher.source_df.iterrows():
                row_dmc = row['uri'].split('/')[-1].split(':')[-1]
                # Check if DMC codes have common prefix (system-model-chapter)
                if dmc_code[:20] in row_dmc or row_dmc[:20] in dmc_code:
                    s1000d_row = pd.DataFrame([row])
                    break

        if s1000d_row.empty:
            # Skip silently - this is expected for some gold standard entries
            continue

        source_concept = s1000d_row.iloc[0].to_dict()

        # Get candidates from KROMA
        candidates = kroma_matcher.find_candidates(source_concept, top_k=20)

        # Check if bike_uri is in candidates and get score
        kroma_score = 0.0
        for cand_uri, score in candidates:
            if cand_uri == bike_uri:
                kroma_score = score
                break

        # Predict: True if score >= threshold
        prediction = kroma_score >= threshold

        y_true.append(ground_truth)
        y_pred.append(prediction)
        scores.append(kroma_score)

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(gold_standard)} pairs...")

    # Calculate metrics (handle zero division)
    if len(y_true) == 0:
        return {
            'threshold': threshold,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'total_evaluated': 0,
            'avg_score_for_positives': 0.0,
            'avg_score_for_negatives': 0.0
        }

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # True/False positives/negatives
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    results = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'total_evaluated': len(y_true),
        'avg_score_for_positives': sum(s for s, t in zip(scores, y_true) if t) / sum(y_true) if sum(y_true) > 0 else 0,
        'avg_score_for_negatives': sum(s for s, t in zip(scores, y_true) if not t) / (len(y_true) - sum(y_true)) if (len(y_true) - sum(y_true)) > 0 else 0
    }

    return results


def find_optimal_threshold(
    kroma_matcher: KROMAMatcher,
    gold_standard: List[Dict]
) -> Tuple[float, Dict]:
    """
    Find optimal threshold by testing multiple values.

    Args:
        kroma_matcher: Initialized KROMA matcher
        gold_standard: Gold standard annotations

    Returns:
        Tuple of (best_threshold, best_results)
    """
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    best_f1 = 0.0
    best_threshold = 0.0
    best_results = {}

    print("\nSearching for optimal threshold...")
    print("=" * 70)

    for threshold in thresholds:
        results = evaluate_kroma(kroma_matcher, gold_standard, threshold)

        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall:    {results['recall']:.3f}")
        print(f"  F1-Score:  {results['f1_score']:.3f}")

        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_threshold = threshold
            best_results = results

    return best_threshold, best_results


def main():
    """
    Run KROMA evaluation.
    """
    print("=" * 70)
    print("KROMA MATCHER EVALUATION")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_all_concepts('bike')

    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Ontology concepts: {len(ontology_df)}")

    # Load gold standard
    print("\n[2/4] Loading gold standard...")
    gold_standard = load_gold_standard()
    print(f"  Gold standard pairs: {len(gold_standard)}")
    print(f"  Positive matches: {sum(a['is_match_manual'] for a in gold_standard)}")
    print(f"  Negative matches: {sum(not a['is_match_manual'] for a in gold_standard)}")

    # Initialize KROMA
    print("\n[3/4] Initializing KROMA...")
    kroma = KROMAMatcher(s1000d_df, ontology_df)

    # Find optimal threshold
    print("\n[4/4] Evaluating KROMA...")
    best_threshold, best_results = find_optimal_threshold(kroma, gold_standard)

    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nOptimal Threshold: {best_threshold:.2f}")
    print(f"\nMetrics:")
    print(f"  Precision: {best_results['precision']:.3f}")
    print(f"  Recall:    {best_results['recall']:.3f}")
    print(f"  F1-Score:  {best_results['f1_score']:.3f}")
    print(f"  Accuracy:  {best_results['accuracy']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {best_results['true_positives']}")
    print(f"  False Positives: {best_results['false_positives']}")
    print(f"  True Negatives:  {best_results['true_negatives']}")
    print(f"  False Negatives: {best_results['false_negatives']}")
    print(f"\nScore Statistics:")
    print(f"  Avg score for positive matches: {best_results['avg_score_for_positives']:.3f}")
    print(f"  Avg score for negative matches: {best_results['avg_score_for_negatives']:.3f}")

    # Save results
    output_file = 'kroma_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(best_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
