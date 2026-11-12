# Supplementary Material

**Paper:** *Causal Edge Effects for Provenance-Based Intrusion Detection: When Correlational Signals Mislead* (ICEICT 2026)

This package archives the implementation and the run records behind the results reported
in the paper. Every file here was produced by the pipeline described below and is included
because it supports a specific claim in the paper.

## 1. Scope

Included: the CI/CA computation code, the validation and ablation scripts, and the summary
records (JSON/CSV) from which the paper's tables, figures and statistics were taken.

Not included: the DARPA Transparent Computing raw audit data (publicly available), the trained
model checkpoints (from the public PIDSMaker release), and the raw per-edge replay dumps
(large intermediate files; their summaries are included).

## 2. Content tree

```
code/
  causal_edge_effect_pipeline.py       CI/CA computation and ranking assembly
  causal_screening_prototype.py        ICE prototype (single-checkpoint screening)
  closed_form_validation.py            closed-form ICE vs. actual-deletion replay
  removal_control_arms.py              top-loss / random / degree-matched / bottom-loss arms
  reference_set_size_sensitivity.py    ADP as a function of |M|
  reference_set_source_spectrum.py     reference sets drawn from different sources
  reference_set_interaction.py         source x size interaction (k=10 cells)
  seed_separation_protocol.py          oracle vs. censored (labels-only-from-past) seeding
  cluster_corrected_statistics.py      run-level statistics and cluster-exact sign test
  run_level_significance_test.py       checkpoint-level and run-level significance
  gate_form_ablation.py                sigmoid gate vs. hard mask over the same support
  gate_hyperparameter_sweep.py         alpha sweep
  reference_contamination_dilution.py  false endpoints entering the CI mean (dilution)
  case_study_extraction.py             top-20 ranking extraction for the case study
  adp_curves.py                        ADP-vs-K curves

results/
  main_adp_table/                      per-checkpoint ADP, all six runs (42 checkpoints)
  cluster_statistics/                  ICC, per-run deltas, sign-test inputs
  causal_necessity_removal/            top-K CI vs. top-K loss removal summary
  reference_set_sensitivity/           |M| grid, source spectrum, interaction analysis
  seed_separation/                     attack-delay cells, six representative checkpoints
  gate_form_ablation/                  CA vs. binary indicator vs. hard mask vs. random gate
  gate_hyperparameter/                 alpha in {0.5,1,2,5}
  reference_contamination/             uniform and top-100 false-positive seeds
  removal_controls/                    seed-level cells, TOST, dose-response matrix
  closed_form_validation/              closed form vs. replay, control envelope
  case_study/                          baseline and gated top-20 rankings (Velox, Orthrus)
```

## 3. Data and models

| Item | Detail |
|---|---|
| Datasets | CADETS_E3 (FreeBSD, three attack scenarios), THEIA_E3 (Linux, two) — DARPA TC Engagement 3 |
| Edge-level models | Orthrus (TGN+GAT), Velox (linear), Kairos (GAT+TGN) |
| Node-level model | Flash (GraphSAGE), used as the transfer target |
| Checkpoints | epochs 7–19 (odd), 42 checkpoints over six runs; representatives ep13 (CADETS), ep19 (THEIA) |
| Reference set | oracle = full ground-truth endpoint set of the test window (72 CADETS / 118 THEIA) |
| Metric | Average Detection Precision (ADP); Precision@K over attack-touching edges |

## 4. Run directory to paper claim

| Directory | Claim in the paper |
|---|---|
| `main_adp_table/` | Table I; 41/42 checkpoints improve |
| `cluster_statistics/` | run-level ICC and cluster-exact sign test (p = 0.031) |
| `causal_necessity_removal/` | Sec. V-B: removing 5 CI edges reduces ADP by 71–91%; top-20 loss removal increases it by up to +141.8% |
| `reference_set_sensitivity/` | Fig. 2 (gain vs. |M|); reference-set source spectrum |
| `seed_separation/` | Sec. V-F: censored variant 18/57 cells; oracle variant 56/57 |
| `gate_form_ablation/` | Sec. V-A: hard mask vs. sigmoid gate; random gate recovers none |
| `gate_hyperparameter/` | Sec. V-A: alpha sweep changes ADP by at most 0.051 |
| `reference_contamination/` | Sec. V-A: true-dilution outcomes for both gate forms |
| `removal_controls/` | Sec. V-F: seed-level equivalence (TOST), dose-response matrix |
| `closed_form_validation/` | Sec. V-F: closed form vs. replay at endpoint level |
| `case_study/` | Fig. 4: ranking inversion, Velox-CADETS_E3 ep13 |

## 5. Reproduction

```
export CAUSAL_WORKSPACE=<path to a PIDSMaker workspace with edge-loss tables>
python3 code/removal_control_arms.py <model> <dataset> <epoch>
python3 code/seed_separation_protocol.py <model> <dataset> <epoch>
python3 code/cluster_corrected_statistics.py
```

Each script documents its inputs and outputs in its header. Scripts expect the workspace
layout used by the public PIDSMaker pipeline (per-window edge-loss CSVs under
`artifacts/training/.../edge_losses/test` and the engagement ground-truth files).

## 6. Notes

- Directory and file names describe what each artifact supports; no internal run identifiers are used.
- Absolute paths are replaced by the `CAUSAL_WORKSPACE` environment variable.
- Numerical values in the result files are unmodified; only names and path references were
  adapted for portability.
