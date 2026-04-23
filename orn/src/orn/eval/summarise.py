"""Collate benchmark JSON files into a paper-ready comparison table.

Reads every *.json under bench/results/ (or an explicit list), picks the
headline metric per task, and produces:

  - a plain-text table for the README / diary
  - a markdown table for the Medium article
  - a LaTeX table stub for the paper

Baseline rows (GPT-2 Small, Pythia-410M, SmolLM2-135M) are hardcoded from
the public Paper's eval harness numbers; the live rows come from our
bench/results/*.json files.
"""
from __future__ import annotations

import json
from pathlib import Path


# Canonical column order for the paper table.
COLUMNS = ["hellaswag", "piqa", "arc_easy", "arc_challenge",
           "winogrande", "boolq", "openbookqa"]

# Metric preference per task (HellaSwag uses acc_norm, ARC uses acc_norm,
# OpenBookQA uses acc_norm; the rest use acc). Matches the lm-eval paper
# conventions so comparisons are apples-to-apples.
METRIC = {
    "hellaswag": "acc_norm",
    "piqa": "acc",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
    "openbookqa": "acc_norm",
}

# Published baseline numbers from lm-eval-harness public leaderboards.
BASELINES = [
    ("GPT-2 Small (124M)",  {"hellaswag": 31.6, "piqa": 62.5, "arc_easy": 43.8,
                               "arc_challenge": 22.7, "winogrande": 51.6,
                               "boolq": 48.8, "openbookqa": 27.4}),
    ("Pythia-160M",         {"hellaswag": 30.1, "piqa": 62.6, "arc_easy": 43.3,
                               "arc_challenge": 22.5, "winogrande": 51.4,
                               "boolq": 55.3, "openbookqa": 27.2}),
    ("Pythia-410M",         {"hellaswag": 40.6, "piqa": 66.7, "arc_easy": 52.1,
                               "arc_challenge": 24.4, "winogrande": 53.8,
                               "boolq": 60.6, "openbookqa": 30.4}),
    ("SmolLM2-135M",        {"hellaswag": 42.1, "piqa": 68.3, "arc_easy": 54.4,
                               "arc_challenge": 30.1, "winogrande": 56.5,
                               "boolq": 60.2, "openbookqa": 34.6}),
]


def _pct(x):
    return "—" if x is None else f"{x*100 if x <= 1 else x:.1f}"


def load_runs(results_dir: str | Path = "bench/results") -> list[dict]:
    runs = []
    for p in sorted(Path(results_dir).glob("*.json")):
        runs.append(json.loads(p.read_text()))
    return runs


def _pick(record: dict, task: str) -> float | None:
    scores = record.get("tasks", {}).get(task, {})
    metric = METRIC.get(task, "acc")
    return scores.get(metric)


def format_text(runs: list[dict]) -> str:
    header = f"{'model':<28} " + "  ".join(f"{c:<10}" for c in COLUMNS)
    lines = [header, "-" * len(header)]
    for name, vals in BASELINES:
        row = f"{name:<28} " + "  ".join(
            f"{_pct(vals.get(c)):<10}" for c in COLUMNS)
        lines.append(row)
    for r in runs:
        label = f"ORN {r['checkpoint']} ({r.get('n_params', 0)/1e6:.0f}M)"
        row = f"{label:<28} " + "  ".join(
            f"{_pct(_pick(r, c)):<10}" for c in COLUMNS)
        lines.append(row)
    return "\n".join(lines)


def format_markdown(runs: list[dict]) -> str:
    cols = COLUMNS
    lines = ["| model | " + " | ".join(cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for name, vals in BASELINES:
        lines.append("| " + name + " | "
                     + " | ".join(_pct(vals.get(c)) for c in cols) + " |")
    for r in runs:
        label = f"**ORN {r['checkpoint']}** ({r.get('n_params', 0)/1e6:.0f}M)"
        lines.append("| " + label + " | "
                     + " | ".join(_pct(_pick(r, c)) for c in cols) + " |")
    return "\n".join(lines)


def format_latex(runs: list[dict]) -> str:
    lines = [r"\begin{tabular}{l" + "r" * len(COLUMNS) + r"}",
             r"\toprule",
             "Model & " + " & ".join(COLUMNS) + r" \\",
             r"\midrule"]
    for name, vals in BASELINES:
        lines.append(name + " & " +
                      " & ".join(_pct(vals.get(c)) for c in COLUMNS) + r" \\")
    lines.append(r"\midrule")
    for r in runs:
        label = f"\\textbf{{ORN {r['checkpoint']}}}"
        lines.append(label + " & " +
                      " & ".join(_pct(_pick(r, c)) for c in COLUMNS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench/results")
    ap.add_argument("--format", choices=["text", "md", "latex", "all"],
                    default="all")
    args = ap.parse_args()
    runs = load_runs(args.dir)
    if args.format in ("text", "all"):
        print("\n=== plain text ===")
        print(format_text(runs))
    if args.format in ("md", "all"):
        print("\n=== markdown ===")
        print(format_markdown(runs))
    if args.format in ("latex", "all"):
        print("\n=== latex ===")
        print(format_latex(runs))


if __name__ == "__main__":
    main()
