"""
Benchmark Evaluation Script for Ontology Matching Pipeline.
Evaluates performance against manually annotated gold standard.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from data_loader import load_all_concepts
from build_knowledge_base import ModernBERTEmbedder, cosine_similarity_batch
from llm_reasoner import LLMReasoner


# Configuration
GOLD_STANDARD_FILE = Path('gold_standard_metrics.json')
BENCHMARK_RESULTS_FILE = Path('benchmark_results.json')
BENCHMARK_REPORT_FILE = Path('benchmark_report.txt')

# Thresholds (same as in build_knowledge_base.py)
HIGH_CONFIDENCE_THRESHOLD = 0.85  # Direct match, no LLM needed
LOW_CONFIDENCE_THRESHOLD = 0.60   # LLM verification needed


def load_gold_standard() -> List[Dict[str, Any]]:
    """
    Load gold standard annotations.
    """
    if not GOLD_STANDARD_FILE.exists():
        raise FileNotFoundError(
            f"Gold standard file not found: {GOLD_STANDARD_FILE}\n"
            "Please run create_gold_standard.py first to create annotations."
        )

    with open(GOLD_STANDARD_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} gold standard annotations")
    return data


def get_concept_embedding(
    uri: str,
    df: pd.DataFrame,
    embeddings: np.ndarray
) -> Tuple[pd.Series, np.ndarray]:
    """
    Get concept and its embedding by URI.
    """
    idx = df[df['uri'] == uri].index[0]
    concept = df.iloc[idx]
    embedding = embeddings[idx]
    return concept, embedding


def evaluate_pair(
    s1000d_concept: pd.Series,
    s1000d_embedding: np.ndarray,
    bike_concept: pd.Series,
    bike_embedding: np.ndarray,
    reasoner: LLMReasoner = None
) -> Dict[str, Any]:
    """
    Evaluate a single S1000D-BikeOntology pair using the pipeline.

    Returns:
        Dictionary with:
            - similarity_score: float
            - method: 'embedding' or 'embedding+llm'
            - is_match_ai: bool (AI decision)
            - llm_confidence: float (if LLM was used)
            - llm_reason: str (if LLM was used)
    """
    # Calculate similarity
    score = float(cosine_similarity_batch(s1000d_embedding, bike_embedding.reshape(1, -1))[0])

    result = {
        'similarity_score': score,
        'method': None,
        'is_match_ai': False,
        'llm_confidence': None,
        'llm_reason': None
    }

    # HIGH CONFIDENCE (>= 0.85): Direct match
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        result['method'] = 'embedding'
        result['is_match_ai'] = True

    # MEDIUM CONFIDENCE (0.60-0.85): LLM verification
    elif score >= LOW_CONFIDENCE_THRESHOLD:
        if reasoner is not None:
            try:
                llm_result = reasoner.verify_match_with_claude(
                    s1000d_concept.to_dict(),
                    bike_concept.to_dict()
                )

                result['method'] = 'embedding+llm'
                result['is_match_ai'] = llm_result['is_match']
                result['llm_confidence'] = llm_result['confidence']
                result['llm_reason'] = llm_result['reason']

            except Exception as e:
                print(f"\n⚠ Warning: LLM verification failed: {e}")
                result['method'] = 'embedding'
                result['is_match_ai'] = False  # Conservative: reject on error
        else:
            result['method'] = 'embedding'
            result['is_match_ai'] = False  # Conservative: reject if no LLM available

    # LOW CONFIDENCE (< 0.60): Reject
    else:
        result['method'] = 'embedding'
        result['is_match_ai'] = False

    return result


def run_benchmark(
    gold_standard: List[Dict[str, Any]],
    s1000d_df: pd.DataFrame,
    ontology_df: pd.DataFrame,
    s1000d_embeddings: np.ndarray,
    ontology_embeddings: np.ndarray,
    reasoner: LLMReasoner = None
) -> List[Dict[str, Any]]:
    """
    Run benchmark evaluation on gold standard.
    """
    print("\n" + "=" * 70)
    print("RUNNING BENCHMARK EVALUATION")
    print("=" * 70)

    results = []
    claude_calls = 0
    claude_accepts = 0
    claude_rejects = 0

    print(f"\nEvaluating {len(gold_standard)} pairs...")

    for item in tqdm(gold_standard, desc="Evaluating pairs", unit="pair"):
        # Get concepts and embeddings
        s1000d_concept, s1000d_embedding = get_concept_embedding(
            item['s1000d_uri'],
            s1000d_df,
            s1000d_embeddings
        )

        bike_concept, bike_embedding = get_concept_embedding(
            item['bike_uri'],
            ontology_df,
            ontology_embeddings
        )

        # Evaluate with pipeline
        eval_result = evaluate_pair(
            s1000d_concept,
            s1000d_embedding,
            bike_concept,
            bike_embedding,
            reasoner
        )

        # Track Claude usage
        if eval_result['method'] == 'embedding+llm':
            claude_calls += 1
            if eval_result['is_match_ai']:
                claude_accepts += 1
            else:
                claude_rejects += 1

        # Combine with gold standard
        result = {
            **item,  # Include original gold standard data
            **eval_result  # Add AI evaluation results
        }

        results.append(result)

    print(f"\n✓ Evaluation complete")
    print(f"\n  Pipeline statistics:")
    print(f"    High confidence matches (≥0.85): {sum(1 for r in results if r['method'] == 'embedding')}")
    print(f"    Claude verifications (0.60-0.85): {claude_calls}")
    print(f"      ✓ Accepted: {claude_accepts}")
    print(f"      ✗ Rejected: {claude_rejects}")

    return results


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.
    """
    # Extract labels
    y_true = [r['is_match_manual'] for r in results]
    y_pred = [r['is_match_ai'] for r in results]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(
            y_true,
            y_pred,
            target_names=['No Match', 'Match'],
            zero_division=0
        )
    }

    # Add confusion matrix breakdown
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics['confusion_matrix_breakdown'] = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

    return metrics


def generate_report(
    metrics: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_file: Path = BENCHMARK_REPORT_FILE
) -> str:
    """
    Generate human-readable benchmark report.
    """
    report_lines = []

    def add_line(text: str = ""):
        report_lines.append(text)

    # Header
    add_line("=" * 70)
    add_line("ONTOLOGY MATCHING BENCHMARK REPORT")
    add_line("=" * 70)
    add_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add_line(f"Gold Standard: {GOLD_STANDARD_FILE}")
    add_line(f"Total Evaluations: {len(results)}")
    add_line()

    # Overall Metrics
    add_line("=" * 70)
    add_line("OVERALL PERFORMANCE METRICS")
    add_line("=" * 70)
    add_line()
    add_line(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    add_line(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    add_line(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    add_line(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    add_line(f"  MCC:       {metrics['mcc']:.4f}")
    add_line()

    # Confusion Matrix
    add_line("=" * 70)
    add_line("CONFUSION MATRIX")
    add_line("=" * 70)
    add_line()

    cm = metrics['confusion_matrix_breakdown']
    add_line("                  Predicted")
    add_line("                  No Match    Match")
    add_line("  Actual  ")
    add_line(f"  No Match       {cm['true_negatives']:6d}     {cm['false_positives']:6d}")
    add_line(f"  Match          {cm['false_negatives']:6d}     {cm['true_positives']:6d}")
    add_line()

    add_line("  Breakdown:")
    add_line(f"    True Positives (TP):  {cm['true_positives']:6d}  (Correctly identified matches)")
    add_line(f"    True Negatives (TN):  {cm['true_negatives']:6d}  (Correctly rejected non-matches)")
    add_line(f"    False Positives (FP): {cm['false_positives']:6d}  (Incorrectly identified as matches)")
    add_line(f"    False Negatives (FN): {cm['false_negatives']:6d}  (Missed actual matches)")
    add_line()

    # Method Breakdown
    add_line("=" * 70)
    add_line("PIPELINE METHOD BREAKDOWN")
    add_line("=" * 70)
    add_line()

    embedding_only = [r for r in results if r['method'] == 'embedding']
    embedding_llm = [r for r in results if r['method'] == 'embedding+llm']

    add_line(f"  High Confidence (≥0.85):    {len(embedding_only):4d} pairs")
    add_line(f"  Claude Verified (0.60-0.85): {len(embedding_llm):4d} pairs")
    add_line()

    # Method-specific metrics
    if embedding_only:
        y_true_emb = [r['is_match_manual'] for r in embedding_only]
        y_pred_emb = [r['is_match_ai'] for r in embedding_only]
        acc_emb = accuracy_score(y_true_emb, y_pred_emb)
        add_line(f"    High Confidence Accuracy: {acc_emb:.4f} ({acc_emb*100:.2f}%)")

    if embedding_llm:
        y_true_llm = [r['is_match_manual'] for r in embedding_llm]
        y_pred_llm = [r['is_match_ai'] for r in embedding_llm]
        acc_llm = accuracy_score(y_true_llm, y_pred_llm)
        add_line(f"    Claude Verified Accuracy: {acc_llm:.4f} ({acc_llm*100:.2f}%)")

    add_line()

    # Detailed Classification Report
    add_line("=" * 70)
    add_line("DETAILED CLASSIFICATION REPORT")
    add_line("=" * 70)
    add_line()
    add_line(metrics['classification_report'])

    # Error Analysis
    add_line("=" * 70)
    add_line("ERROR ANALYSIS")
    add_line("=" * 70)
    add_line()

    # False Positives
    false_positives = [r for r in results if not r['is_match_manual'] and r['is_match_ai']]
    add_line(f"  False Positives: {len(false_positives)}")
    if false_positives:
        add_line("\n  Top 3 False Positives (by similarity score):")
        sorted_fp = sorted(false_positives, key=lambda x: x['similarity_score'], reverse=True)
        for i, fp in enumerate(sorted_fp[:3], 1):
            add_line(f"\n    {i}. Score: {fp['similarity_score']:.3f} | Method: {fp['method']}")
            add_line(f"       S1000D: {fp['s1000d_label']}")
            add_line(f"       Bike:   {fp['bike_label']}")

    # False Negatives
    add_line()
    false_negatives = [r for r in results if r['is_match_manual'] and not r['is_match_ai']]
    add_line(f"  False Negatives: {len(false_negatives)}")
    if false_negatives:
        add_line("\n  Top 3 False Negatives (by similarity score):")
        sorted_fn = sorted(false_negatives, key=lambda x: x['similarity_score'], reverse=True)
        for i, fn in enumerate(sorted_fn[:3], 1):
            add_line(f"\n    {i}. Score: {fn['similarity_score']:.3f} | Method: {fn['method']}")
            add_line(f"       S1000D: {fn['s1000d_label']}")
            add_line(f"       Bike:   {fn['bike_label']}")

    add_line()
    add_line("=" * 70)
    add_line("END OF REPORT")
    add_line("=" * 70)

    report_text = "\n".join(report_lines)

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text


def main():
    """
    Main benchmark execution.
    """
    print("=" * 70)
    print("ONTOLOGY MATCHING BENCHMARK")
    print("=" * 70)

    # Step 1: Load gold standard
    print("\n[1/5] LOADING GOLD STANDARD")
    print("-" * 70)
    gold_standard = load_gold_standard()

    # Step 2: Load data and embeddings
    print("\n[2/5] LOADING DATA & EMBEDDINGS")
    print("-" * 70)

    df = load_all_concepts(
        s1000d_folder='bike',
        include_ontologies=True
    )

    s1000d_df = df[df['source'] == 's1000d'].reset_index(drop=True)
    ontology_df = df[df['source'] == 'bike_ontology'].reset_index(drop=True)

    print(f"\n✓ Data loaded:")
    print(f"  S1000D concepts: {len(s1000d_df)}")
    print(f"  Ontology concepts: {len(ontology_df)}")

    # Step 3: Build embeddings
    print("\n[3/5] BUILDING EMBEDDINGS")
    print("-" * 70)

    embedder = ModernBERTEmbedder()

    # Check for cache
    from build_knowledge_base import EMBEDDING_CACHE_FILE
    import pickle

    if EMBEDDING_CACHE_FILE.exists():
        print(f"✓ Loading cached embeddings from {EMBEDDING_CACHE_FILE}")
        with open(EMBEDDING_CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        all_embeddings = cache['embeddings']
    else:
        print("Building embeddings (this will take a few minutes)...")
        all_texts = []
        for _, row in df.iterrows():
            label = row.get('label', '')
            context = row.get('context', '')
            if label and context:
                combined = f"{label}. {context[:500]}"
            elif label:
                combined = label
            else:
                combined = context[:500] if context else ""
            all_texts.append(combined)

        all_embeddings = embedder.embed_batch(all_texts, show_progress=True, batch_size=8)

    # Split embeddings
    s1000d_indices = df[df['source'] == 's1000d'].index.tolist()
    ontology_indices = df[df['source'] == 'bike_ontology'].index.tolist()

    s1000d_embeddings = all_embeddings[s1000d_indices]
    ontology_embeddings = all_embeddings[ontology_indices]

    print(f"\n✓ Embeddings ready")

    # Step 4: Initialize LLM reasoner
    print("\n[4/5] INITIALIZING CLAUDE API")
    print("-" * 70)

    try:
        reasoner = LLMReasoner()
        print("✓ Claude API initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize Claude API: {e}")
        print("  Proceeding without LLM verification (may reduce accuracy)")
        reasoner = None

    # Step 5: Run benchmark
    print("\n[5/5] RUNNING BENCHMARK")
    print("-" * 70)

    results = run_benchmark(
        gold_standard,
        s1000d_df,
        ontology_df,
        s1000d_embeddings,
        ontology_embeddings,
        reasoner
    )

    # Save results
    print(f"\nSaving results to {BENCHMARK_RESULTS_FILE}...")
    with open(BENCHMARK_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)

    # Generate report
    print(f"\nGenerating report to {BENCHMARK_REPORT_FILE}...")
    report = generate_report(metrics, results)

    # Print report to console
    print("\n" * 2)
    print(report)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)
    print(f"\n  Results saved to: {BENCHMARK_RESULTS_FILE}")
    print(f"  Report saved to:  {BENCHMARK_REPORT_FILE}")


if __name__ == '__main__':
    main()
