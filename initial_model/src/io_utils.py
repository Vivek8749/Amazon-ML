"""Output formatting and I/O utilities."""
import os
import pandas as pd


def write_matching_results(matches: dict, output_path: str):
    """Write matching_results.tsv in the required format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for s1_id in sorted(matches.keys()):
        matched = matches[s1_id]
        matched_str = ",".join(sorted(set(matched))) if matched else ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"[Output] Wrote {len(df)} rows to {output_path}")


def write_candidate_pairs(candidates: dict, output_path: str):
    """Write candidate_pairs.tsv in the required format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for s1_id in sorted(candidates.keys()):
        cands = candidates[s1_id]
        cands_str = ",".join(sorted(set(cands))) if cands else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cands_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"[Output] Wrote {len(df)} rows to {output_path}")


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load ground truth TSV."""
    return pd.read_csv(path, sep="\t", dtype=str)
