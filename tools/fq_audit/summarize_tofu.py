#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = [
    "forget_quality",
    "model_utility",
    "forget_Q_A_gibberish",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "extraction_strength",
    "privleak",
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_forget_truth_ratio(summary_path: Path):
    eval_path = summary_path.with_name("TOFU_EVAL.json")
    if not eval_path.exists():
        return ""
    try:
        logs = read_json(eval_path)
    except json.JSONDecodeError:
        return ""
    values = logs.get("forget_truth_ratio", {}).get("value_by_index", {})
    return len(values) if isinstance(values, dict) else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/baldro_fq_audit")
    parser.add_argument("--out", default="results/baldro_fq_audit/tofu_summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for summary_path in sorted(root.rglob("TOFU_SUMMARY.json")):
        try:
            summary = read_json(summary_path)
        except json.JSONDecodeError:
            continue
        row = {
            "summary_path": str(summary_path),
            "run_dir": str(summary_path.parent),
            "forget_truth_ratio_count": count_forget_truth_ratio(summary_path),
        }
        for key in METRIC_KEYS:
            row[key] = summary.get(key, "")
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_dir", "summary_path", *METRIC_KEYS, "forget_truth_ratio_count"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} TOFU summaries under {root}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
