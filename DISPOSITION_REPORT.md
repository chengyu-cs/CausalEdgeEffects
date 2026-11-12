# Disposition Report — Supplementary Material

Scope: selection, anonymization and naming cleanup of the experiment artifacts for the
supplementary package. Every decision is listed with its reason.

## 1. Included (62 files, ~0.8 MB)

Kept: the code that computes CI/CA and every run record that backs a number, table, figure or
statistical claim in the paper (see README §4 for the directory-to-claim mapping), plus the
case-study rankings behind Fig. 4.

## 2. Excluded, with reasons

| Item | Reason |
|---|---|
| Raw run logs (`*.log`, ~20 files) | Machine bookkeeping; contain interpreter paths and per-run echoes. All evidential value is in the summary JSON/CSV that are included. |
| Replay intermediates (`e1a`, `e2f`, `e3`, `e25`; ~3.9 GB) | Per-edge replay dumps retained for re-analysis, not needed to support a reported number. Their summaries are included under `closed_form_validation/` and `removal_controls/`. |
| Buggy prototype implementation and its smoke logs | Superseded implementation; the corrected computation is what produced the reported numbers. |
| Internal planning documents (setup guide, research plan, review-disposition notes) | Not part of the paper's evidence; contain internal process notes and institutional detail. |
| Raw DARPA TC audit data | Publicly available from the original source; too large to redistribute. |
| `pipeline_correction_old_vs_new.csv` | A correction-audit table with columns `old_buggy` / `new_fixed`. Not referenced by the paper. **Excluded for your review** — say the word and it can be added back. |
| `source_spectrum_summary.json` | File self-marks an interpretability hint as "SUPERSEDED … overturned" by a later same-granularity recalculation. The current conclusion is carried by `source_interaction.json` and `source_spectrum_grid.csv`, both included. **Excluded for your review.** |

## 3. Changes made, and how each was verified

| Change | Scope | Verification |
|---|---|---|
| Functional renaming (directory = what it supports, file = what it is) | All 15 scripts, 10 result directories, 49 result files | Filename scan: zero occurrences of experiment-number prefixes |
| Absolute path removal | Script headers, usage lines, output-directory constants | grep for absolute home paths, cloud-host names and interpreter paths: zero hits. Paths replaced by `${WORKSPACE:-.}` / `CAUSAL_WORKSPACE` |
| Internal round identifiers stripped from comments | Script headers and inline notes | Regex scan for internal identifiers: zero hits |
| Numeric content | Result files | Copied byte-for-byte; no value was edited, rounded or recomputed |

## 4. Exemptions

Implementation notes that explain why the second-maximum computation uses `nlargest(2)` rather
than an `nth(1)` aggregation are retained. They document the correctness of the included
implementation and are standard research-code commentary, not process bookkeeping.

## 5. Residual scan

- Identity strings (author names, institutions, e-mail addresses, repository URLs): none.
- Absolute paths or user names: none.
- Internal identifiers (`EXP-`, experiment numbers, review-round labels): none.
- Placeholder markers in code and text: none.
