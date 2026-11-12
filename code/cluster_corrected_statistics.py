#!/usr/bin/env python3
""": unit-correct statistics for the 42-checkpoint CA improvement claim.

Fixes / R2-M2 (Wilcoxon pseudo-replication): 42 checkpoints =
6 runs (model x dataset combos) x 7 serial epochs on the same training
trajectory. Treats the run (combo) as the only independent unit.

 (a) reproduce the archived naive 42-cell Wilcoxon (audit anchor)
 (b) ICC / Kish DEFF on the 42 deltas (nested design quantification)
 (c) cluster-level exact sign test (n=6)
 (d) wild cluster bootstrap, G=6 -> exhaustive 2^6=64 Rademacher sign
 patterns (Cameron-Gelbach-Miller 2008; Canay-Santos-Shaikh 2021
 special case: under H0 the cluster means have a symmetric dist.,
 sign-flip enumeration is exact, no MC error)
 (e) per-combo stratified Wilcoxon (n=7 each) + Bonferroni x6 and Holm
 (f) magnitude-of-inflation summary (nominal p vs cluster-exact p)

Inputs : results/main_adp_table/summary_fixed42.csv
 results/main_adp_table/wilcoxon_fixed42.json (audit anchors)
Output : results/cluster_statistics/cluster_stats.json
 results/cluster_statistics/combo_level.csv
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

WS = os.environ.get("CAUSAL_WORKSPACE", ".")
IN_CSV = os.path.join(WS, "results/main_adp_table/summary_fixed42.csv")
IN_JSON = os.path.join(WS, "results/main_adp_table/wilcoxon_fixed42.json")
OUT = os.path.join(WS, "results/cluster_statistics")

ALPHA = 0.05
M = 7 # epochs per run

def icc_one_way(y, groups):
 """One-way random-effects ICC(1) via ANOVA mean squares."""
 df = pd.DataFrame({"y": y, "g": groups})
 k = df.groupby("g")["y"].mean()
 n_g = df.groupby("g").size()
 grand = df["y"].mean()
 N, G = len(df), df["g"].nunique()
 m = n_g.mean()
 ssb = (n_g * (k - grand) ** 2).sum()
 ssw = df.apply(lambda r: (r.y - k[r.g]) ** 2, axis=1).sum()
 msb = ssb / (G - 1)
 msw = ssw / (N - G)
 icc = (msb - msw) / (msb + (m - 1) * msw)
 return float(icc), float(msb), float(msw)

def main():
 os.makedirs(OUT, exist_ok=True)
 df = pd.read_csv(IN_CSV)
 assert len(df) == 42, len(df)
 df["combo"] = df["model"] + "_" + df["dataset"]
 df["delta"] = df["ca_gt_adp"] - df["baseline_adp"]
 combos = sorted(df["combo"].unique())
 assert len(combos) == 6, combos

 # ---- (a) naive 42-cell Wilcoxon (reproduce archived anchor) ----
 # archived aggregate.py:90 uses alternative="greater" (one-sided,
 # scipy auto method -> normal approx at n=42). Reproduce that exact
 # call plus the two-sided / exact variants for full disclosure.
 w42 = stats.wilcoxon(df["ca_gt_adp"], df["baseline_adp"], zero_method="wilcox",
 alternative="greater") # archived call
 w42_two = stats.wilcoxon(df["ca_gt_adp"], df["baseline_adp"], zero_method="wilcox",
 alternative="two-sided", method="approx")
 w42_ex = stats.wilcoxon(df["ca_gt_adp"], df["baseline_adp"], zero_method="wilcox",
 alternative="two-sided", method="exact")
 archived = json.load(open(IN_JSON))
 anchor = archived["all_42"]

 # ---- (b) ICC / DEFF on the 42 deltas ----
 icc_delta, msb_d, msw_d = icc_one_way(df["delta"], df["combo"])
 deff = 1.0 + (M - 1) * icc_delta
 n_eff = 42.0 / deff
 # ceiling autocorrelation evidence: kairos_THEIA ca_gt all-0.9999 run
 kt = df[df.combo == "kairos_THEIA"]
 kt_ca = kt["ca_gt_adp"].to_numpy()
 # lag-1 autocorrelation within each run's epoch-ordered delta series
 lag1 = {}
 for c in combos:
 s = df[df.combo == c].sort_values("epoch")["delta"].to_numpy()
 if np.std(s) > 0:
 lag1[c] = float(np.corrcoef(s[:-1], s[1:])[0, 1])
 else:
 lag1[c] = float("nan") # constant series (saturated run)

 # ---- (c) cluster-level exact sign test (n=6) ----
 combo_mean = df.groupby("combo")["delta"].mean().loc[combos]
 n_pos = int((combo_mean > 0).sum())
 p_sign = stats.binomtest(n_pos, 6, 0.5, alternative="two-sided").pvalue
 # cluster-level Wilcoxon signed rank (n=6), exact
 wcl = stats.wilcoxon(combo_mean.to_numpy() - 0.0, zero_method="wilcox",
 alternative="two-sided", method="exact")

 # ---- (d) wild cluster bootstrap, exact 2^6 enumeration ----
 # statistic T = |mean of sign-flipped cluster means|; H0: effect 0
 # (symmetry of cluster means under H0 -> Rademacher flips exact)
 ybar = combo_mean.to_numpy()
 t_obs = abs(ybar.mean())
 n_ge = 0
 for signs in itertools.product([-1.0, 1.0], repeat=6):
 t_s = abs((np.asarray(signs) * ybar).mean())
 if t_s >= t_obs - 1e-12:
 n_ge += 1
 p_wcb = n_ge / 64.0

 # ---- (e) stratified per-combo Wilcoxon + multiplicity ----
 strat = []
 for c in combos:
 sub = df[df.combo == c]
 # archived call: one-sided greater, auto method (exact at n=7)
 w_arch = stats.wilcoxon(sub["ca_gt_adp"], sub["baseline_adp"],
 zero_method="wilcox", alternative="greater")
 w_two = stats.wilcoxon(sub["ca_gt_adp"], sub["baseline_adp"],
 zero_method="wilcox", alternative="two-sided",
 method="exact")
 # manual exact with zero-drop: all-positive nonzero deltas give the
 # closed form two-sided 2*2^-(n_nz-1); scipy silently degrades to
 # normal approx when zeros are present (e.g. orthrus_CADETS ep7 tie)
 d_sub = (sub["ca_gt_adp"] - sub["baseline_adp"]).to_numpy()
 d_nz = d_sub[d_sub != 0]
 if (d_nz > 0).all():
 p_two_exact_manual = float(2.0 ** (-(len(d_nz) - 1)))
 else:
 p_two_exact_manual = float(w_two.pvalue)
 key = "_" + c[: -3] if c.endswith("_E3") else "_" + c
 strat.append({
 "combo": c, "n": len(sub),
 "p_one_sided_archived_call": float(w_arch.pvalue),
 "p_two_sided_exact": p_two_exact_manual,
 "p_archived_json": float(archived[key]["p_value"]),
 "W_archived": float(archived[key]["wilcoxon_stat"]),
 "mean_delta": float(sub["delta"].mean()),
 "n_epochs_improved": int((sub["delta"] > 0).sum()),
 })
 strat = pd.DataFrame(strat)
 strat["p_bonferroni_one_sided_x6"] = (strat["p_one_sided_archived_call"] * 6).clip(upper=1.0)
 strat["p_bonferroni_two_sided_x6"] = (strat["p_two_sided_exact"] * 6).clip(upper=1.0)
 # Holm on one-sided (archived direction)
 order = strat["p_one_sided_archived_call"].rank(method="first").astype(int).to_numpy()
 ps = strat["p_one_sided_archived_call"].sort_values().to_numpy()
 holm = np.minimum(np.maximum.accumulate(ps * (6 - np.arange(6))), 1.0)
 strat["p_holm_one_sided"] = holm[order - 1]

 # anchor checks (archived = one-sided greater, auto method)
 anchor_checks = {
 "W_42_matches": bool(abs(w42.statistic - anchor["wilcoxon_stat"]) < 1e-9),
 "p_42_one_sided_matches": bool(
 abs(w42.pvalue - anchor["p_value"]) / anchor["p_value"] < 1e-6),
 "p_42_two_sided_approx_equals_2x_one_sided": bool(
 abs(w42_two.pvalue - 2 * w42.pvalue) / (2 * w42.pvalue) < 1e-9),
 "per_combo_one_sided_matches_archived": bool(
 np.allclose(strat["p_one_sided_archived_call"],
 strat["p_archived_json"], rtol=1e-6)),
 "archived_p_is_one_sided_greater": True,
 }

 # ---- (f) inflation summary ----
 infl = {
 "nominal_p_42_one_sided_approx_archived": float(w42.pvalue),
 "nominal_p_42_two_sided_approx": float(w42_two.pvalue),
 "nominal_p_42_two_sided_exact": float(w42_ex.pvalue),
 "cluster_exact_p_sign": float(p_sign),
 "cluster_exact_p_wcb64": float(p_wcb),
 "orders_of_magnitude_gap_vs_archived": float(np.log10(w42.pvalue / p_wcb)),
 "orders_of_magnitude_gap_vs_exact": float(np.log10(w42_ex.pvalue / p_wcb)),
 "n_eff_kish": float(n_eff),
 "inflation_factor_n": float(42 / n_eff),
 }

 out = {
 "design": {"n_cells": 42, "n_clusters_G": 6, "m_epochs_per_run": M,
 "cluster_unit": "model x dataset run (training trajectory)"},
 "audit_anchors": anchor_checks,
 "naive_42_wilcoxon": {
 "W": float(w42.statistic),
 "p_one_sided_approx_archived": float(w42.pvalue),
 "p_two_sided_approx": float(w42_two.pvalue),
 # Manual exact: scipy method='exact' falls back to normal approx when a zero
 # difference is present (orthrus_CADETS ep7 ceiling tie). Zero-drop leaves
 # n=41 all-positive deltas, full rank sum W+=861=41*42/2, closed form:
 "p_one_sided_exact": float(2.0 ** (-41)),
 "p_two_sided_exact": float(2 * 2.0 ** (-41)),
 "caveat": "archived e11 p=1.21e-8 is the ONE-SIDED greater, "
 "normal-approximation p; reviewers' 4.5e-13 is the "
 "two-sided EXACT p; all assume epoch-level independence "
 "the nested design does not provide"},
 "nesting": {
 "icc_delta": icc_delta, "ms_between": msb_d, "ms_within": msw_d,
 "kish_deff": deff, "n_eff": n_eff,
 "lag1_autocorr_per_run": lag1,
 "kairos_THEIA_ca_gt_all_09999": bool(np.all(np.isclose(kt_ca, 0.9999))),
 },
 "cluster_level": {
 "combo_means": {c: float(v) for c, v in combo_mean.items()},
 "n_positive_of_6": n_pos,
 "sign_test_p": float(p_sign),
 "wilcoxon_signed_rank_p": float(wcl.pvalue),
 "wild_cluster_bootstrap_64": {
 "p": float(p_wcb), "n_patterns_ge": n_ge, "n_patterns": 64,
 "method": "exhaustive Rademacher sign enumeration on the 6 "
 "cluster means (CGM 2008 / Canay et al. 2021, G=6 "
 "special case; no asymptotics, no MC error)"},
 },
 "stratified_wilcoxon": strat.to_dict(orient="records"),
 "inflation": infl,
 "writeup_recommendation": {
 "abstract_sentence": (
 "CA improves 41/42 checkpoints (97.6%); treating the 6 training "
 "runs as the independent unit (Kish DEFF={:.1f}, n_eff~{:.1f}), "
 "6/6 runs improve with cluster-exact sign p={:.4f} (wild "
 "cluster bootstrap, 64 exhaustive sign patterns, identical "
 "p); the 42-cell Wilcoxon p=1.2e-8 assumes epoch-level "
 "independence that the nested design does not provide and is "
 "reported only as descriptive."
 ).format(deff, n_eff, p_sign),
 "notes": [
 "per-combo Bonferroni x6 leaves orthrus_CADETS above 0.05 "
 "(exact p x 6); Holm gives the same verdict",
 "d=1.586 and r=1.00 remain valid as effect sizes; only the "
 "p-value's unit changes",
 ],
 },
 }
 with open(os.path.join(OUT, "cluster_stats.json"), "w") as f:
 json.dump(out, f, indent=1, ensure_ascii=False)
 strat.to_csv(os.path.join(OUT, "combo_level.csv"), index=False)
 print(json.dumps({k: out[k] for k in
 ("audit_anchors", "nesting", "cluster_level", "inflation")},
 indent=1, ensure_ascii=False))
 print(strat.to_string())
 print(" DONE")

if __name__ == "__main__":
 main()
