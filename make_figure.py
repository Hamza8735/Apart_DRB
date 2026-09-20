"""Figure 1: DRB does not replicate on frontier models, and the models discriminate.

Two panels, both reproducible from out/graded.jsonl (+ data/tasks.yaml for DRB's
reported per-category rates):

  (a) Non-replication. Per DRB task category, refusal rate: DRB reported vs this
      work (pooled over models, seed A). DRB's spread collapses to our ~0 floor.
  (b) Discrimination. Per model, refusal on the DEFENSIVE grid (near 0, with 95%
      scenario-cluster-bootstrap CI) vs the OFFENSIVE should-refuse anchor
      (near 1, with coverage bounds over unresolved completions). Dashed line =
      DRB's reported overall 0.122.

Colours are the dataviz reference categorical palette (blue slot 1, orange slot
2), validated CVD-safe; DRB/prior is recessive gray + hatch so identity survives
greyscale printing. One value axis per panel; no dual axes.

Usage: python scripts/make_figure.py
Out:   out/figure1.png (300 dpi) and out/figure1.pdf
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260913)
B = 5000

REFUSAL = {"full_refusal", "partial_refusal"}
NON_MODEL = {"filtered", "no_response", "truncated", "UNPARSED"}
DRB_OVERALL = 0.122

# dataviz reference palette (validated): slot1 blue, slot2 orange; gray = prior.
C_OURS = "#2a78d6"      # this work / defensive
C_ANCHOR = "#eb6834"    # offensive anchor
C_DRB = "#b6b5ad"       # DRB / prior (recessive)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2dd"

MODEL_LABEL = {
    "anthropic/claude-sonnet-5": "Claude\nSonnet 5",
    "z-ai/glm-5.2": "GLM 5.2",
    "~openai/gpt-sol-latest": "GPT 5.6\n(Sol)",
}
MODEL_ORDER = ["anthropic/claude-sonnet-5", "z-ai/glm-5.2", "~openai/gpt-sol-latest"]


def load():
    rows = [json.loads(l) for l in
            (ROOT / "out" / "graded.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    tasks = yaml.safe_load((ROOT / "data" / "tasks.yaml").read_text(encoding="utf-8"))
    drb = {t["drb_category"]: t["drb_reported_refusal"] for t in tasks}
    return rows, drb


def is_ref(r):
    return r["judge_class"] in REFUSAL


def grid_seedA(rows):
    return [r for r in rows if not r.get("is_anchor")
            and r["judge_class"] not in NON_MODEL and r.get("seed_tag", "A") == "A"]


def cluster_ci(sub, stat):
    """95% CI resampling scenarios with replacement."""
    if not sub:
        return (np.nan, np.nan, np.nan)
    by = {}
    for r in sub:
        by.setdefault(r["scenario_id"], []).append(r)
    scen = list(by)
    point = stat(sub)
    draws = np.empty(B)
    for i in range(B):
        pick = RNG.choice(scen, size=len(scen), replace=True)
        boot = [r for s in pick for r in by[s]]
        draws[i] = stat(boot)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def rate_ref(sub):
    return float(np.mean([is_ref(r) for r in sub])) if sub else np.nan


def panel_a(ax, rows, drb):
    grid = grid_seedA(rows)
    cats = sorted(drb, key=lambda c: drb[c], reverse=True)
    ours = []
    for c in cats:
        sub = [r for r in grid if r["drb_category"] == c]
        ours.append(rate_ref(sub) if sub else np.nan)
    y = np.arange(len(cats))
    h = 0.38
    ax.barh(y + h / 2 + 0.01, [drb[c] for c in cats], height=h, color=C_DRB,
            hatch="////", edgecolor="white", linewidth=0.6, label="DRB (reported)")
    ax.barh(y - h / 2 - 0.01, ours, height=h, color=C_OURS,
            edgecolor="white", linewidth=0.6, label="This work")
    for yi, c in zip(y, cats):
        ax.text(drb[c] + 0.006, yi + h / 2 + 0.01, f"{drb[c]:.2f}",
                va="center", ha="left", fontsize=7.2, color=INK2)
        o = ours[cats.index(c)]
        ax.text(max(o, 0) + 0.006, yi - h / 2 - 0.01, f"{o:.3f}",
                va="center", ha="left", fontsize=7.2, color=C_OURS, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in cats], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.52)
    ax.set_xlabel("Refusal rate", fontsize=9, color=INK2)
    ax.set_title("(a) The DRB effect does not replicate", fontsize=10.5,
                 color=INK, loc="left", pad=8, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=8.2)
    _despine(ax, xgrid=True)


def panel_b(ax, rows):
    grid = grid_seedA(rows)
    anchor = [r for r in rows if r.get("is_anchor")]
    x = np.arange(len(MODEL_ORDER))
    w = 0.38
    def_pt, def_lo, def_hi, an_pt, an_lo, an_hi, ann = [], [], [], [], [], [], []
    for m in MODEL_ORDER:
        gsub = [r for r in grid if r["model"] == m]
        p, lo, hi = cluster_ci(gsub, rate_ref)
        def_pt.append(p); def_lo.append(p - lo); def_hi.append(hi - p)
        gg = [r for r in anchor if r["model"] == m]
        res = [r for r in gg if r["judge_class"] not in NON_MODEL]
        n_all, n_res = len(gg), len(res)
        ref = sum(is_ref(r) for r in res)
        rate = ref / n_res if n_res else np.nan
        blo, bhi = (ref / n_all, (ref + (n_all - n_res)) / n_all) if n_all else (np.nan, np.nan)
        an_pt.append(rate); an_lo.append(rate - blo); an_hi.append(bhi - rate)
        ann.append(f"{ref}/{n_res}" + (f"\nof {n_all}" if n_all != n_res else ""))
    ax.bar(x - w / 2 - 0.01, def_pt, width=w, color=C_OURS, edgecolor="white",
           linewidth=0.6, label="Defensive prompts (this work)",
           yerr=[def_lo, def_hi], error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
    ax.bar(x + w / 2 + 0.01, an_pt, width=w, color=C_ANCHOR, edgecolor="white",
           linewidth=0.6, hatch="\\\\\\", label="Offensive anchor (HarmBench)",
           yerr=[an_lo, an_hi], error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
    for xi, p in zip(x, def_pt):
        ax.text(xi - w / 2 - 0.01, p + 0.03, f"{p:.3f}", ha="center", va="bottom",
                fontsize=7.6, color=C_OURS, fontweight="bold")
    for xi, p, hi, a in zip(x, an_pt, an_hi, ann):
        ax.text(xi + w / 2 + 0.01, min(p + hi + 0.02, 1.02), f"{p:.2f}",
                ha="center", va="bottom", fontsize=7.8, color=INK, fontweight="bold")
        ax.text(xi + w / 2 + 0.01, 0.02, a, ha="center", va="bottom",
                fontsize=6.6, color=INK2)
    ax.axhline(DRB_OVERALL, ls=(0, (4, 3)), lw=1.2, color=INK2, zorder=1)
    ax.text(len(MODEL_ORDER) - 0.5, DRB_OVERALL + 0.02,
            f"DRB reported overall ({DRB_OVERALL:.3f})", ha="right", va="bottom",
            fontsize=7.6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Refusal rate", fontsize=9, color=INK2)
    ax.set_title("(b) Same models refuse offensive prompts", fontsize=10.5,
                 color=INK, loc="left", pad=8, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=8.2, ncol=1,
              bbox_to_anchor=(0.0, 0.98))
    _despine(ax, ygrid=True)


def _despine(ax, xgrid=False, ygrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def main():
    rows, drb = load()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.patch.set_facecolor("white")
    panel_a(axa, rows, drb)
    panel_b(axb, rows)
    fig.tight_layout(w_pad=3)
    out_png = ROOT / "out" / "figure1.png"
    out_pdf = ROOT / "out" / "figure1.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"-> {out_png}\n-> {out_pdf}")


if __name__ == "__main__":
    main()
