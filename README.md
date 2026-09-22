# Defensive Refusal Bias on Frontier Models

A controlled replication and extension of **Defensive Refusal Bias (DRB)**
(Campbell et al., 2026) on 2026-era models. DRB found that safety-tuned LLMs
refuse *legitimate* cyber-defence requests when those requests carry offensive
vocabulary or authorisation framing. We test whether that reproduces on current
models using a controlled design that holds the technical task fixed and varies
only surface form.

**Headline result.** On a task set of clearly-defensive incident-response
prompts, the effect does **not** reproduce: overall refusal is **0.005** vs
DRB's reported **0.122**, and it is *measurement-independent* (an LLM judge and
DRB's own regex detector agree the rate is at floor). Because refusal is at
floor, the terminology/authorisation contrasts are underpowered and are **not**
interpreted as evidence of no effect. A held-out should-refuse anchor confirms
the models still refuse offensive requests, so the null reflects **calibration,
not blanket compliance**.

Research conducted at the Apart Research AI Incident Response Sprint,
September 2026.

## Repository layout

```
scripts/
  build_prompts.py   expand the task set into the full prompt grid
  run.py             query models via OpenRouter (provider-pinned, resumable)
  grade.py           four-class LLM-judge grading + refusal-regex baseline
  analyse.py         rates, scenario-cluster bootstrap CIs, contrasts, anchor
  calibrate.py       label-permutation check on the inference procedure
  kappa.py           judge-vs-hand-grade agreement (Cohen's kappa)
  handgrade_tui.py   terminal tool for blind hand-grading
  make_figure.py     the headline figure
data/
  tasks.yaml         12 defensive scenarios (terminology pairs + elements)
  anchor.yaml        held-out should-refuse prompts (see Data & ethics)
out/                 generated locally; git-ignored
```

## Reproduce

```bash
python -m pip install -r requirements.txt
cp .env.example .env        # add your OpenRouter key
export OPENROUTER_API_KEY=sk-or-...   # or rely on .env

python scripts/build_prompts.py                                   # -> data/prompts.jsonl
python scripts/run.py   --model anthropic/claude-sonnet-5 --rollouts 5
python scripts/run.py   --model <other models...>        --rollouts 5
python scripts/grade.py --judge <judge-model>                     # -> out/graded.jsonl
python scripts/analyse.py                                         # rates + contrasts + anchor
python scripts/make_figure.py                                     # -> out/figure1.png / .pdf
```

Judge validation (optional but reported):

```bash
python scripts/grade.py --sample-for-handgrade 40 --population anchor
python scripts/handgrade_tui.py --population anchor
python scripts/kappa.py --population anchor
```

Runs are resumable and additive: re-running skips
`(cell_id, model, rollout, seed_tag)` keys already present, so you can add a
model or a seed without regrading existing rows.

## Method notes

- **Unit of inference is the scenario.** Rollouts share a prompt and cells share
  a scenario, so all confidence intervals come from a bootstrap that resamples
  scenarios (12 clusters — reported as a limitation).
- **Provider is pinned** (no router fallback) so a moderation refusal from the
  serving stack can be told apart from a refusal by the model itself.
- **Non-model outcomes are recorded, not dropped.** Empty completions, provider
  content-filtering and truncations are excluded from rates and reported as
  coverage, so denominators stay auditable.
- **Four-class taxonomy** (full compliance / degraded / partial refusal / full
  refusal) separates degraded assistance from refusal — impairment a binary
  refusal metric misses.

## Data & ethics

The should-refuse anchor is drawn from HarmBench's published cybercrime-intrusion
set and is used only as an aggregate validity check. **Model completions to the
anchor prompts are not released**, and no generated outputs are committed to this
repository. Run the pipeline locally to regenerate them.

## Limitations

Refusal at floor leaves the moderator contrasts underpowered; the four-class
taxonomy's degraded tier is near-empty on the frontier models; the seed-null
replicate covers one model; 12 scenario clusters give a small bootstrap; and the
design (controlled) is confounded with model generation (2026 vs the earlier
models DRB evaluated). See the write-up for details.

## Write-up

The full report is in [`writeup/`](writeup/) (add your PDF here).

## Citation

Builds on Campbell et al. (2026), *Defensive Refusal Bias: How Safety Alignment
Fails Cyber Defenders* (arXiv:2603.01246).

## LLM usage

Claude (via Claude Code) assisted with planning, code, and analysis tooling. The
report was written by the author.
