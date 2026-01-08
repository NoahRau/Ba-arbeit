"""
Evaluate Annotated Matches.

Reads manually annotated matches and computes comprehensive metrics:
- Precision, Recall, F1-Score
- Matthews Correlation Coefficient (MCC)
- Accuracy
- Confusion Matrix
- Per-matcher breakdown

Generates detailed evaluation report.
"""

import json
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


def load_annotated_data(csv_file: str) -> pd.DataFrame:
    """
    Load annotated CSV file.

    Args:
        csv_file: Path to annotated CSV

    Returns:
        DataFrame with annotations
    """
    df = pd.read_csv(csv_file, encoding='utf-8')

    # Validate required columns
    required_cols = ['s1000d_uri', 'pipeline_selected_uri', 'is_match_manual']

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert is_match_manual to boolean
    df['is_match_manual'] = df['is_match_manual'].astype(str).str.upper()
    df['is_match_manual'] = df['is_match_manual'].map({
        'TRUE': True,
        'YES': True,
        '1': True,
        'FALSE': False,
        'NO': False,
        '0': False,
        'NULL': False,
        '': False
    })

    # Fill NaN with False
    df['is_match_manual'] = df['is_match_manual'].fillna(False)

    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        df: DataFrame with annotations

    Returns:
        Dictionary with metrics
    """
    # Ground truth
    y_true = df['is_match_manual'].values

    # Pipeline predictions
    # TRUE if pipeline found a match (not NULL)
    y_pred = (df['pipeline_selected_uri'] != 'NULL').values

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

    metrics = {
        'total_samples': len(df),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'mcc': float(mcc),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,

        # Additional stats
        'ground_truth_positives': int(y_true.sum()),
        'ground_truth_negatives': int((~y_true).sum()),
        'predicted_positives': int(y_pred.sum()),
        'predicted_negatives': int((~y_pred).sum())
    }

    return metrics


def generate_report(
    df: pd.DataFrame,
    metrics: dict,
    output_file: str = None
) -> str:
    """
    Generate detailed evaluation report as Markdown.

    Args:
        df: Annotated DataFrame
        metrics: Computed metrics
        output_file: Output markdown file path

    Returns:
        Report text
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_report_{timestamp}.md"

    # Build report
    report_lines = []

    report_lines.append("# Hybrid Pipeline Evaluation Report")
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
    report_lines.append("")

    # Interpretation
    report_lines.append("### Interpretation")
    report_lines.append("")
    report_lines.append(f"- **Precision ({metrics['precision']*100:.1f}%):** Of all matches the pipeline found, {metrics['precision']*100:.1f}% were correct.")
    report_lines.append(f"- **Recall ({metrics['recall']*100:.1f}%):** Of all true matches, the pipeline found {metrics['recall']*100:.1f}%.")
    report_lines.append(f"- **F1-Score ({metrics['f1_score']*100:.1f}%):** Harmonic mean of precision and recall.")
    report_lines.append(f"- **MCC ({metrics['mcc']:.3f}):** Correlation between predictions and ground truth (-1 to +1, 0 is random).")
    report_lines.append("")

    # Confusion Matrix
    report_lines.append("## Confusion Matrix")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append("                 Predicted")
    report_lines.append("                 No Match    Match")
    report_lines.append("Actual  ")
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

    # Detailed Breakdown
    report_lines.append("## Detailed Breakdown")
    report_lines.append("")

    # Pipeline methods
    if 'pipeline_method' in df.columns:
        method_counts = df['pipeline_method'].value_counts()
        report_lines.append("### Pipeline Methods Used")
        report_lines.append("")
        report_lines.append("| Method | Count |")
        report_lines.append("|--------|-------|")
        for method, count in method_counts.items():
            report_lines.append(f"| {method} | {count} |")
        report_lines.append("")

        # Accuracy by method
        report_lines.append("### Accuracy by Pipeline Method")
        report_lines.append("")
        report_lines.append("| Method | Correct | Total | Accuracy |")
        report_lines.append("|--------|---------|-------|----------|")

        for method in method_counts.index:
            method_df = df[df['pipeline_method'] == method]
            y_true_method = method_df['is_match_manual'].values
            y_pred_method = (method_df['pipeline_selected_uri'] != 'NULL').values

            correct = (y_true_method == y_pred_method).sum()
            total = len(method_df)
            acc = correct / total if total > 0 else 0

            report_lines.append(f"| {method} | {correct} | {total} | {acc:.2%} |")

        report_lines.append("")

    # Classification Report
    report_lines.append("## Sklearn Classification Report")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(metrics['classification_report'])
    report_lines.append("```")
    report_lines.append("")

    # Error Analysis
    report_lines.append("## Error Analysis")
    report_lines.append("")

    # False Positives
    false_positives = df[(df['pipeline_selected_uri'] != 'NULL') & (~df['is_match_manual'])]
    if len(false_positives) > 0:
        report_lines.append(f"### False Positives ({len(false_positives)})")
        report_lines.append("")
        report_lines.append("Pipeline found a match, but it was incorrect:")
        report_lines.append("")

        for i, row in false_positives.head(10).iterrows():
            report_lines.append(f"**{i+1}. {row['s1000d_label']}**")
            report_lines.append(f"- Pipeline selected: {row.get('selected_label', 'N/A')}")
            report_lines.append(f"- Confidence: {row.get('pipeline_confidence', 0):.3f}")
            report_lines.append(f"- Method: {row.get('pipeline_method', 'N/A')}")
            report_lines.append("")

        if len(false_positives) > 10:
            report_lines.append(f"*... and {len(false_positives) - 10} more*")
            report_lines.append("")

    # False Negatives
    false_negatives = df[(df['pipeline_selected_uri'] == 'NULL') & (df['is_match_manual'])]
    if len(false_negatives) > 0:
        report_lines.append(f"### False Negatives ({len(false_negatives)})")
        report_lines.append("")
        report_lines.append("Pipeline didn't find a match, but there should be one:")
        report_lines.append("")

        for i, row in false_negatives.head(10).iterrows():
            report_lines.append(f"**{i+1}. {row['s1000d_label']}**")
            report_lines.append(f"- Correct match: {row.get('correct_match_uri', 'Not specified')}")
            report_lines.append(f"- Notes: {row.get('notes', 'N/A')}")
            report_lines.append("")

        if len(false_negatives) > 10:
            report_lines.append(f"*... and {len(false_negatives) - 10} more*")
            report_lines.append("")

    # Summary
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"The hybrid pipeline achieved:")
    report_lines.append(f"- **F1-Score: {metrics['f1_score']:.2%}**")
    report_lines.append(f"- **Precision: {metrics['precision']:.2%}**")
    report_lines.append(f"- **Recall: {metrics['recall']:.2%}**")
    report_lines.append(f"- **MCC: {metrics['mcc']:.3f}**")
    report_lines.append("")

    if metrics['f1_score'] >= 0.85:
        report_lines.append("✅ **Excellent performance!** The pipeline meets state-of-the-art standards.")
    elif metrics['f1_score'] >= 0.70:
        report_lines.append("✓ **Good performance.** The pipeline is working well.")
    elif metrics['f1_score'] >= 0.50:
        report_lines.append("⚠ **Moderate performance.** Consider tuning matcher weights or improving LLM prompts.")
    else:
        report_lines.append("❌ **Poor performance.** Significant improvements needed.")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*Generated by Hybrid Ontology Matching Pipeline*")

    # Write report
    report_text = "\n".join(report_lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text, output_file


def plot_confusion_matrix(metrics: dict, output_file: str = None):
    """
    Plot confusion matrix visualization.

    Args:
        metrics: Metrics dictionary
        output_file: Output image file path
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"confusion_matrix_{timestamp}.png"

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
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Hybrid Pipeline')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Confusion matrix plot saved: {output_file}")


def main():
    """Run evaluation on annotated data."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate annotated matches')
    parser.add_argument('csv_file', type=str, help='Annotated CSV file')
    parser.add_argument('--output', type=str, help='Output report file (markdown)')
    parser.add_argument('--plot', action='store_true', help='Generate confusion matrix plot')

    args = parser.parse_args()

    print("=" * 70)
    print("HYBRID PIPELINE EVALUATION")
    print("=" * 70)

    # Load data
    print(f"\n[1/4] Loading annotated data from: {args.csv_file}")
    df = load_annotated_data(args.csv_file)
    print(f"  ✓ Loaded {len(df)} annotated samples")

    # Compute metrics
    print("\n[2/4] Computing metrics...")
    metrics = compute_metrics(df)
    print(f"  ✓ F1-Score: {metrics['f1_score']:.4f}")
    print(f"  ✓ MCC: {metrics['mcc']:.4f}")

    # Generate report
    print("\n[3/4] Generating evaluation report...")
    report_text, report_file = generate_report(df, metrics, args.output)
    print(f"  ✓ Report saved: {report_file}")

    # Save metrics as JSON
    metrics_file = report_file.replace('.md', '_metrics.json')
    with open(metrics_file, 'w') as f:
        # Convert numpy types to native Python
        metrics_json = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                       for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)
    print(f"  ✓ Metrics JSON saved: {metrics_file}")

    # Plot confusion matrix
    if args.plot:
        print("\n[4/4] Generating confusion matrix plot...")
        plot_confusion_matrix(metrics)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nPrecision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"MCC:       {metrics['mcc']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    print(f"  FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
