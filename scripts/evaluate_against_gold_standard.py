"""
Evaluate Pipeline Results against Gold Standard.

Compares pipeline matches with manually annotated gold standard
and computes comprehensive evaluation metrics.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR


def load_gold_standard(gold_file: str) -> pd.DataFrame:
    """
    Load gold standard annotations from JSON.

    Args:
        gold_file: Path to gold_standard.json

    Returns:
        DataFrame with gold standard annotations
    """
    with open(gold_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])

    # Convert to DataFrame
    gold_df = pd.DataFrame(annotations)

    print(f"✓ Loaded {len(gold_df)} gold standard annotations")

    return gold_df


def load_pipeline_results(csv_file: str) -> pd.DataFrame:
    """
    Load pipeline results from CSV.

    Args:
        csv_file: Path to pipeline CSV

    Returns:
        DataFrame with pipeline results
    """
    df = pd.read_csv(csv_file, encoding='utf-8')

    print(f"✓ Loaded {len(df)} pipeline results")

    return df


def merge_gold_and_pipeline(gold_df: pd.DataFrame, pipeline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge gold standard with pipeline results.

    Args:
        gold_df: Gold standard DataFrame
        pipeline_df: Pipeline results DataFrame

    Returns:
        Merged DataFrame
    """
    # Merge on source URI
    merged = pd.merge(
        gold_df,
        pipeline_df,
        left_on='source_uri',
        right_on='s1000d_uri',
        how='inner',
        suffixes=('_gold', '_pipeline')
    )

    print(f"✓ Merged {len(merged)} concepts (found in both gold standard and pipeline)")

    if len(merged) == 0:
        print("\n⚠️  WARNING: No matching concepts found!")
        print("   Check that URIs match between gold standard and pipeline CSV")

    return merged


def compute_metrics(merged_df: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics.

    Args:
        merged_df: Merged DataFrame with gold and pipeline results

    Returns:
        Dictionary with metrics
    """
    # Ground truth from gold standard
    y_true = merged_df['is_match'].values  # Gold standard: is there a match?

    # Pipeline predictions
    y_pred = (merged_df['pipeline_selected_uri'] != 'NULL').values  # Pipeline: did it find a match?

    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Classification report
    class_report = classification_report(
        y_true, y_pred,
        target_names=['No Match', 'Match'],
        zero_division=0
    )

    # URI-level accuracy (for matches only)
    matches_df = merged_df[merged_df['is_match'] == True]
    if len(matches_df) > 0:
        # Handle both possible column names (with and without suffix)
        correct_uri_col = 'correct_match_uri' if 'correct_match_uri' in matches_df.columns else 'correct_match_uri_gold'
        if correct_uri_col in matches_df.columns:
            correct_uris = (matches_df[correct_uri_col] == matches_df['pipeline_selected_uri']).sum()
            uri_accuracy = correct_uris / len(matches_df)
        else:
            uri_accuracy = 0.0
    else:
        uri_accuracy = 0.0

    metrics = {
        'total_samples': len(merged_df),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'mcc': float(mcc),
        'uri_accuracy': float(uri_accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'gold_positives': int(y_true.sum()),
        'gold_negatives': int((~y_true).sum()),
        'predicted_positives': int(y_pred.sum()),
        'predicted_negatives': int((~y_pred).sum())
    }

    return metrics


def generate_report(
    merged_df: pd.DataFrame,
    metrics: dict,
    output_file: str = None
) -> str:
    """
    Generate detailed evaluation report.

    Args:
        merged_df: Merged DataFrame
        metrics: Computed metrics
        output_file: Output markdown file path

    Returns:
        Report text
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"gold_standard_evaluation_{timestamp}.md"

    report_lines = []

    report_lines.append("# Gold Standard Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Total Samples:** {metrics['total_samples']}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Main Metrics
    report_lines.append("## Main Metrics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| **Precision** | {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%) |")
    report_lines.append(f"| **Recall** | {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%) |")
    report_lines.append(f"| **F1-Score** | {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%) |")
    report_lines.append(f"| **Accuracy** | {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) |")
    report_lines.append(f"| **MCC (Matthews)** | {metrics['mcc']:.4f} |")
    report_lines.append(f"| **URI Accuracy** | {metrics['uri_accuracy']:.4f} ({metrics['uri_accuracy']*100:.2f}%) |")
    report_lines.append("")

    # Interpretation
    report_lines.append("### Interpretation")
    report_lines.append("")
    report_lines.append(f"- **Precision ({metrics['precision']*100:.1f}%):** Of all matches the pipeline found, {metrics['precision']*100:.1f}% were correct.")
    report_lines.append(f"- **Recall ({metrics['recall']*100:.1f}%):** Of all true matches (gold standard), the pipeline found {metrics['recall']*100:.1f}%.")
    report_lines.append(f"- **F1-Score ({metrics['f1_score']*100:.1f}%):** Harmonic mean of precision and recall.")
    report_lines.append(f"- **MCC ({metrics['mcc']:.3f}):** Correlation between predictions and ground truth (-1 to +1, 0 is random).")
    report_lines.append(f"- **URI Accuracy ({metrics['uri_accuracy']*100:.1f}%):** For matches, {metrics['uri_accuracy']*100:.1f}% selected the correct target URI.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Confusion Matrix
    report_lines.append("## Confusion Matrix")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append("                 Predicted")
    report_lines.append("                 No Match    Match")
    report_lines.append("Actual")
    report_lines.append(f"No Match    {metrics['true_negatives']:>8}  {metrics['false_positives']:>8}")
    report_lines.append(f"Match       {metrics['false_negatives']:>8}  {metrics['true_positives']:>8}")
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("| Metric | Count |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| True Positives (TP) | {metrics['true_positives']} |")
    report_lines.append(f"| False Positives (FP) | {metrics['false_positives']} |")
    report_lines.append(f"| True Negatives (TN) | {metrics['true_negatives']} |")
    report_lines.append(f"| False Negatives (FN) | {metrics['false_negatives']} |")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Classification Report
    report_lines.append("## Classification Report")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(metrics['classification_report'])
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Error Analysis
    report_lines.append("## Error Analysis")
    report_lines.append("")

    # False Positives
    false_positives = merged_df[
        (merged_df['pipeline_selected_uri'] != 'NULL') &
        (merged_df['is_match'] == False)
    ]

    if len(false_positives) > 0:
        report_lines.append(f"### False Positives ({len(false_positives)})")
        report_lines.append("")
        report_lines.append("Pipeline found a match, but gold standard says NO MATCH:")
        report_lines.append("")

        for i, (idx, row) in enumerate(false_positives.head(10).iterrows(), 1):
            report_lines.append(f"**{i}. {row['source_label']}**")
            report_lines.append(f"- Pipeline selected: {row.get('selected_label', 'N/A')}")
            report_lines.append(f"- Pipeline confidence: {row.get('pipeline_confidence', 0):.3f}")
            report_lines.append(f"- Gold standard: NULL (no match)")
            if pd.notna(row.get('notes')):
                report_lines.append(f"- Notes: {row['notes']}")
            report_lines.append("")

        if len(false_positives) > 10:
            report_lines.append(f"*... and {len(false_positives) - 10} more*")
            report_lines.append("")

    # False Negatives
    false_negatives = merged_df[
        (merged_df['pipeline_selected_uri'] == 'NULL') &
        (merged_df['is_match'] == True)
    ]

    if len(false_negatives) > 0:
        report_lines.append(f"### False Negatives ({len(false_negatives)})")
        report_lines.append("")
        report_lines.append("Pipeline said NO MATCH, but gold standard has a match:")
        report_lines.append("")

        # Determine correct URI column name
        correct_uri_col_fn = 'correct_match_uri' if 'correct_match_uri' in merged_df.columns else 'correct_match_uri_gold'

        for i, (idx, row) in enumerate(false_negatives.head(10).iterrows(), 1):
            report_lines.append(f"**{i}. {row['source_label']}**")
            report_lines.append(f"- Gold standard match: {row.get(correct_uri_col_fn, 'N/A')}")
            if pd.notna(row.get('notes')):
                report_lines.append(f"- Notes: {row['notes']}")
            report_lines.append("")

        if len(false_negatives) > 10:
            report_lines.append(f"*... and {len(false_negatives) - 10} more*")
            report_lines.append("")

    # Wrong URI matches (TP but wrong URI)
    # Determine correct URI column name
    correct_uri_col = 'correct_match_uri' if 'correct_match_uri' in merged_df.columns else 'correct_match_uri_gold'

    if correct_uri_col in merged_df.columns:
        wrong_uris = merged_df[
            (merged_df['is_match'] == True) &
            (merged_df['pipeline_selected_uri'] != 'NULL') &
            (merged_df[correct_uri_col] != merged_df['pipeline_selected_uri'])
        ]
    else:
        wrong_uris = pd.DataFrame()  # Empty if column doesn't exist

    if len(wrong_uris) > 0:
        report_lines.append(f"### Wrong URI Matches ({len(wrong_uris)})")
        report_lines.append("")
        report_lines.append("Pipeline found a match, but selected the WRONG target URI:")
        report_lines.append("")

        for i, (idx, row) in enumerate(wrong_uris.head(10).iterrows(), 1):
            report_lines.append(f"**{i}. {row['source_label']}**")
            report_lines.append(f"- Pipeline selected: {row.get('selected_label', 'N/A')}")
            report_lines.append(f"- Gold standard correct: {row.get(correct_uri_col, 'N/A')}")
            if pd.notna(row.get('notes')):
                report_lines.append(f"- Notes: {row['notes']}")
            report_lines.append("")

    # Summary
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"The pipeline achieved against the gold standard:")
    report_lines.append(f"- **F1-Score: {metrics['f1_score']:.2%}**")
    report_lines.append(f"- **Precision: {metrics['precision']:.2%}**")
    report_lines.append(f"- **Recall: {metrics['recall']:.2%}**")
    report_lines.append(f"- **MCC: {metrics['mcc']:.3f}**")
    report_lines.append(f"- **URI Accuracy: {metrics['uri_accuracy']:.2%}**")
    report_lines.append("")

    if metrics['f1_score'] >= 0.85:
        report_lines.append("✅ **Excellent performance!** The pipeline meets state-of-the-art standards.")
    elif metrics['f1_score'] >= 0.70:
        report_lines.append("✓ **Good performance.** The pipeline is working well.")
    elif metrics['f1_score'] >= 0.50:
        report_lines.append("⚠ **Moderate performance.** Consider tuning parameters.")
    else:
        report_lines.append("❌ **Poor performance.** Significant improvements needed.")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*Generated by Gold Standard Evaluation Script*")

    # Write report
    report_text = "\n".join(report_lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text, output_file


def plot_confusion_matrix(metrics: dict, output_file: str = None):
    """
    Plot confusion matrix.

    Args:
        metrics: Metrics dictionary
        output_file: Output image file path
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"confusion_matrix_{timestamp}.png"

    cm = np.array(metrics['confusion_matrix'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Match', 'Match'],
        yticklabels=['No Match', 'Match']
    )
    plt.ylabel('Gold Standard (Actual)')
    plt.xlabel('Pipeline (Predicted)')
    plt.title('Confusion Matrix - Pipeline vs Gold Standard')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Confusion matrix saved: {output_file}")


def main():
    """Run evaluation against gold standard."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate pipeline against gold standard')
    parser.add_argument('gold_standard', type=str, help='Path to gold_standard.json')
    parser.add_argument('pipeline_csv', type=str, help='Path to pipeline CSV')
    parser.add_argument('--output', type=str, help='Output report file (markdown)')
    parser.add_argument('--plot', action='store_true', help='Generate confusion matrix plot')

    args = parser.parse_args()

    print("=" * 70)
    print("GOLD STANDARD EVALUATION")
    print("=" * 70)

    # Load data
    print(f"\n[1/5] Loading gold standard: {args.gold_standard}")
    gold_df = load_gold_standard(args.gold_standard)

    print(f"\n[2/5] Loading pipeline results: {args.pipeline_csv}")
    pipeline_df = load_pipeline_results(args.pipeline_csv)

    # Merge
    print(f"\n[3/5] Merging gold standard with pipeline results...")
    merged_df = merge_gold_and_pipeline(gold_df, pipeline_df)

    if len(merged_df) == 0:
        print("\n❌ ERROR: Cannot evaluate - no matching concepts found!")
        return

    # Compute metrics
    print("\n[4/5] Computing metrics...")
    metrics = compute_metrics(merged_df)
    print(f"  ✓ F1-Score: {metrics['f1_score']:.4f}")
    print(f"  ✓ MCC: {metrics['mcc']:.4f}")

    # Generate report
    print("\n[5/5] Generating evaluation report...")
    report_text, report_file = generate_report(merged_df, metrics, args.output)
    print(f"  ✓ Report saved: {report_file}")

    # Save metrics as JSON
    metrics_file = str(report_file).replace('.md', '_metrics.json')
    with open(metrics_file, 'w') as f:
        metrics_json = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                       for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)
    print(f"  ✓ Metrics JSON saved: {metrics_file}")

    # Plot confusion matrix
    if args.plot:
        print("\nGenerating confusion matrix plot...")
        plot_confusion_matrix(metrics)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nPrecision:     {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:      {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"MCC:           {metrics['mcc']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"URI Accuracy:  {metrics['uri_accuracy']:.4f} ({metrics['uri_accuracy']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    print(f"  FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
