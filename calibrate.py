"""Calibration check: does the inference procedure hold its error rate?

From the secret-loyalties audit, which checked that its p-values were uniform
under a true null (organism c). Here the analogue: shuffle the framing labels
WITHIN each scenario, so any real framing effect is destroyed, then run the
exact same significance procedure. Under this null the procedure should call
'significant' about 5% of the time. If it fires much more often, your cluster
bootstrap is anti-conservative at 12 clusters and your p-values need a caveat.

This converts "my intervals might be too narrow with only 12 clusters" from a
reviewer's objection into a reported number.

Usage: python scripts/calibrate.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(4242)
N_SHUFFLE = 200      # null replicates
B = 300              # bootstrap reps per replicate (kept modest for speed)

REFUSAL = {"full_refusal", "partial_refusal"}
IMPAIRED = {"full_refusal", "partial_refusal", "degraded"}


def load_grid() -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in
                       (ROOT / "out" / "graded.jsonl").read_text().splitlines() if l.strip()])
    if "is_anchor" in df.columns:
        df = df[~df.is_anchor.fillna(False)]
    df = df[df.judge_class != "UNPARSED"]
    df = df[df.seed_tag == "A"]
    df["is_impaired"] = df.judge_class.isin(IMPAIRED)
    return df


def contrast_sig(df: pd.DataFrame, a: str, b: str, key: str = "framing") -> bool:
    """True if the a-minus-b contrast on is_impaired excludes zero (95% cluster boot)."""
    scen = df.scenario_id.unique()
    groups = {s: g for s, g in df.groupby("scenario_id")}

    def stat(d):
        ga, gb = d[d[key] == a], d[d[key] == b]
        if not len(ga) or not len(gb):
            return np.nan
        return ga.is_impaired.mean() - gb.is_impaired.mean()

    draws = np.empty(B)
    for i in range(B):
        pick = RNG.choice(scen, len(scen), replace=True)
        draws[i] = stat(pd.concat([groups[s] for s in pick], ignore_index=True))
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return not (lo <= 0 <= hi)


def main() -> None:
    df = load_grid()
    print(f"grid rows (seed A, parsed): {len(df)}  scenarios: {df.scenario_id.nunique()}")
    print(f"shuffling framing labels within scenario, {N_SHUFFLE} times\n")

    for a, b in [("casual", "bare"), ("structured", "bare")]:
        fires = 0
        for _ in range(N_SHUFFLE):
            sh = df.copy()
            # permute framing within each scenario -> destroys any real effect
            sh["framing"] = (sh.groupby("scenario_id")["framing"]
                             .transform(lambda x: RNG.permutation(x.values)))
            if contrast_sig(sh, a, b):
                fires += 1
        rate = fires / N_SHUFFLE
        flag = "OK" if rate <= 0.10 else "ANTI-CONSERVATIVE - caveat your p-values"
        print(f"{a:>10} - bare : fires {fires}/{N_SHUFFLE} = {rate:.3f} "
              f"(nominal 0.05)  {flag}")

    print("\nReport the realised rate in your methods section. A rate near 0.05 "
          "says\nthe 12-cluster bootstrap is well calibrated here; higher says "
          "treat the\nframing p-values as approximate.")


if __name__ == "__main__":
    main()
