"""Macro-averaged F-beta (beta=0.5) evaluation, matching the leaderboard metric,
plus a breakdown of where the remaining loss comes from."""
import numpy as np

from .data import parse_ground_truth

# ===== EVALUATION ==============================================================

def _f05(pred, truth, beta=0.5):
    if not truth and not pred: return 1.0
    if not truth and pred:     return 0.0
    if truth and not pred:     return 0.0
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    if p+r == 0: return 0.0
    return (1+beta**2)*p*r / (beta**2*p + r)


def macro_f05(predictions, truth):
    """Macro F0.5 over the entities in `truth` ({s1_id: set of true ids})."""
    if not truth:
        return 0.0
    return float(np.mean([_f05(set(predictions.get(sid, [])), t) for sid, t in truth.items()]))


def evaluate(predictions, gt_df):
    gt = parse_ground_truth(gt_df)
    scores = [_f05(set(predictions.get(sid, [])), truth) for sid, truth in gt.items()]
    macro = np.mean(scores) if scores else 0.0
    print(f"[Eval] Macro F_0.5 = {macro:.4f}  ({len(scores):,} entities)")
    return macro


def loss_breakdown(predictions, truth, candidates=None):
    """Where the macro-F0.5 loss (1 - score) comes from, per error type.

    Each entity loses 1 - F0.5; the categories partition the entities, so
    their losses add up to the total. With `candidates`, also reports the
    blocking ceiling: the score of an oracle that picks exactly the true
    matches present in the candidates.
    """
    cats = {
        "singleton given matches": [0, 0.0],   # truth empty, prediction not
        "matched entity left empty": [0, 0.0],  # truth not empty, prediction empty
        "wrong extra IDs": [0, 0.0],            # both non-empty, some predicted IDs wrong
        "missing IDs only": [0, 0.0],           # both non-empty, all predicted right, some missed
    }
    n = len(truth)
    for sid, t in truth.items():
        pred = set(predictions.get(sid, []))
        loss = 1.0 - _f05(pred, t)
        if loss == 0:
            continue
        if not t:
            key = "singleton given matches"
        elif not pred:
            key = "matched entity left empty"
        elif pred - t:
            key = "wrong extra IDs"
        else:
            key = "missing IDs only"
        cats[key][0] += 1
        cats[key][1] += loss
    total = sum(v[1] for v in cats.values())
    score = 1.0 - total / max(n, 1)
    print(f"\n[Loss] macro F0.5 = {score:.4f}; loss = {total / max(n, 1):.4f} over {n:,} entities")
    for name, (cnt, loss) in cats.items():
        share = loss / total if total else 0.0
        print(f"  {name:28s} {cnt:7,} entities  loss {loss / max(n, 1):.4f}  ({share:5.1%} of loss)")
    report = {"score": score, "categories": {k: {"entities": c, "loss": l / max(n, 1)}
                                             for k, (c, l) in cats.items()}}
    if candidates is not None:
        oracle = {sid: sorted(t & set(candidates.get(sid, []))) for sid, t in truth.items()}
        ceiling = macro_f05(oracle, truth)
        print(f"  blocking ceiling (perfect model on these candidates): {ceiling:.4f} "
              f"— {1 - ceiling:.4f} of loss is unrecoverable without better blocking")
        report["blocking_ceiling"] = ceiling
    return report
