"""
Evaluation utilities — macro-averaged F_0.5 scoring.
"""
import pandas as pd
import numpy as np


def compute_f_beta_per_entity(predicted: set, truth: set, beta: float = 0.5) -> float:
    """Compute F_beta for a single entity."""
    # Singleton handling
    if not truth and not predicted:
        return 1.0  # Correctly predicted singleton
    if not truth and predicted:
        return 0.0  # False merge on singleton
    if truth and not predicted:
        return 0.0  # Missed all matches

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return f_beta


def evaluate(predictions: dict, ground_truth: pd.DataFrame, beta: float = 0.5) -> dict:
    """
    Compute macro-averaged F_beta score.
    
    Args:
        predictions: {s1_entity_id: [list of matched entity_ids]}
        ground_truth: DataFrame with source1_entity_id, matched_entity_ids columns
        beta: beta parameter for F score (default 0.5)
    
    Returns:
        dict with f_beta, precision, recall, and per-entity details
    """
    gt_lookup = {}
    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if pd.isna(matched) or matched == "":
            gt_lookup[s1_id] = set()
        else:
            gt_lookup[s1_id] = set(str(matched).split(","))

    scores = []
    details = []

    for s1_id, truth in gt_lookup.items():
        predicted = set(predictions.get(s1_id, []))
        f = compute_f_beta_per_entity(predicted, truth, beta)
        scores.append(f)
        details.append({
            "entity_id": s1_id,
            "f_beta": f,
            "predicted": len(predicted),
            "truth": len(truth),
            "tp": len(predicted & truth),
            "fp": len(predicted - truth),
            "fn": len(truth - predicted),
        })

    macro_f = np.mean(scores) if scores else 0.0

    # Compute overall precision/recall
    total_tp = sum(d["tp"] for d in details)
    total_fp = sum(d["fp"] for d in details)
    total_fn = sum(d["fn"] for d in details)

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    result = {
        "f_beta": macro_f,
        "overall_precision": overall_p,
        "overall_recall": overall_r,
        "n_entities": len(scores),
        "n_with_matches": sum(1 for d in details if d["truth"] > 0),
        "n_singletons": sum(1 for d in details if d["truth"] == 0),
        "n_correct_singletons": sum(1 for d in details if d["truth"] == 0 and d["predicted"] == 0),
    }

    print(f"[Eval] Macro F_{beta}: {macro_f:.4f}")
    print(f"[Eval] Overall Precision: {overall_p:.4f}, Recall: {overall_r:.4f}")
    print(f"[Eval] Entities: {result['n_entities']}, "
          f"With matches: {result['n_with_matches']}, "
          f"Singletons: {result['n_singletons']} "
          f"(correct: {result['n_correct_singletons']})")

    return result
