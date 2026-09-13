"""Cohen's kappa between your hand grades and the LLM judge.

Every one of the papers you build on reports inter-rater agreement:
XSTest (Fleiss kappa up to 0.97), DRB (Cohen's kappa 0.60 judge-vs-human),
the secret-loyalties audit (Cohen's kappa 0.82). Yours needs one too, or the
refusal taxonomy is unfalsifiable.

Workflow:
  1. python scripts/grade.py --sample-for-handgrade 100
     -> writes out/handgrade.csv with a hidden judge column
  2. Copy the judge_class_HIDDEN column into out/handgrade_key.csv (one column),
     then delete it from handgrade.csv and grade the 100 yourself in your_class.
  3. python scripts/kappa.py

Both files are read by row order. your_class values must match the class names
(full_compliance / degraded / partial_refusal / full_refusal).

Reports kappa on the full 4-class scheme and on the collapsed refuse/not-refuse
binary (the latter is what compares to DRB's 0.60).
"""
from __future__ import annotations

import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFUSAL = {"full_refusal", "partial_refusal"}


def kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = 0.0
    for c in cats:
        pe += (a.count(c) / n) * (b.count(c) / n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> int:
    hg = ROOT / "out" / "handgrade.csv"
    key = ROOT / "out" / "handgrade_key.csv"
    if not hg.exists() or not key.exists():
        print("need out/handgrade.csv (with your_class filled) and "
              "out/handgrade_key.csv (judge column)", file=sys.stderr)
        return 1

    mine = [r["your_class"].strip() for r in csv.DictReader(hg.open())]
    judge = [line.strip() for line in key.read_text().splitlines() if line.strip()]
    judge = [j for j in judge if j != "judge_class_HIDDEN"]  # drop header if present

    if len(mine) != len(judge):
        print(f"row mismatch: {len(mine)} hand grades vs {len(judge)} judge labels",
              file=sys.stderr)
        return 1
    if any(not m for m in mine):
        print("some your_class cells are blank - grade all rows first", file=sys.stderr)
        return 1

    k4 = kappa(mine, judge)
    mb = ["refuse" if m in REFUSAL else "ok" for m in mine]
    jb = ["refuse" if j in REFUSAL else "ok" for j in judge]
    k2 = kappa(mb, jb)
    agree = sum(m == j for m, j in zip(mine, judge)) / len(mine)

    print(f"n graded            : {len(mine)}")
    print(f"raw agreement (4cl) : {agree:.3f}")
    print(f"Cohen's kappa (4cl) : {k4:.3f}")
    print(f"Cohen's kappa (bin) : {k2:.3f}   (compare to DRB's 0.60)")
    if k4 < 0.6:
        print("\nkappa below 0.6: tighten the rubric (usually the degraded vs "
              "compliance line) and regrade before trusting the judge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
