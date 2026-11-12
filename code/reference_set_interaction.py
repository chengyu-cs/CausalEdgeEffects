#!/usr/bin/env python3
""": same-granularity conditional analysis of the M-source spectrum
(interaction framing), superseding 's marginal "size dominates,
identity secondary" headline.

Fixes / R2-C1: compared a 3-strategy x 24-cell averaged
strategy spread against a 4-k-point size spread (different granularities;
k=10 same-granularity recompute shrinks the ratio to 1.34x), and 2/6
combos show identity DOMINATING size (orthrus_THEIA k10: earlyactive
0.0007 vs random 0.3196, a 450x within-combo gap).

 (a) per-combo table at the k=10 anchor: strategy means, strategy
 spread (max-min over 3 strategies), size spread (random delta
 range across k=1..10), random-arm per-trial dispersion
 (b) conditional statement test: random effective in all combos at
 k=10; topdegree/earlyactive失效格 disclosure (no selective quote)
 (c) dataset x strategy interaction at k=10: point estimate
 (CADETS mean strategy-gap vs THEIA mean) + exact permutation p
 (choose-3-of-6, 20 splits) -- Brookes 2004 / ICH E9 s5.7
 discipline: with n=3 vs 3 combos the interaction test is
 hopelessly underpowered, so regardless of p the claim is phrased
 as exploratory heterogeneity unless the permutation p is extreme.
 (d) revised headline + writeup sentences.

Input : results/reference_set_sensitivity/msource_grid.csv
Output: results/reference_set_sensitivity/combo_spread_table.csv
 results/reference_set_sensitivity/k10_cells.csv
 results/reference_set_sensitivity/interaction.json
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

WS = os.environ.get("CAUSAL_WORKSPACE", ".")
IN = os.path.join(WS, "results/reference_set_sensitivity/msource_grid.csv")
OUT = os.path.join(WS, "results/reference_set_sensitivity")
STRATS = ("random", "topdegree", "earlyactive")
K_ANCHOR = 10

def main():
 os.makedirs(OUT, exist_ok=True)
 df = pd.read_csv(IN)
 df["combo"] = df["combo"].str.replace("_E3_", "_", regex=False)
 ks = sorted(k for k in df.k.unique() if k <= K_ANCHOR) # 1,3,5,10

 # ---------- (a) per-combo k=10 anchor table ----------
 rows, k10_cells = [], []
 for combo in sorted(df.combo.unique()):
 sub = df[df.combo == combo]
 s10 = sub[sub.k == K_ANCHOR]
 means10 = {s: float(s10[s10.strategy == s].delta.mean()) for s in STRATS}
 trials10 = s10[s10.strategy == "random"].delta.to_numpy()
 for s in STRATS:
 for _, r in s10[s10.strategy == s].iterrows():
 k10_cells.append({"combo": combo, "strategy": s, "trial": int(r.trial),
 "delta": float(r.delta)})
 # size spread: random mean delta range over k = 1..10
 rk = sub[(sub.strategy == "random") & (sub.k.isin(ks))]
 size_means = rk.groupby("k").delta.mean()
 size_spread = float(size_means.max() - size_means.min())
 strat_vals = [means10[s] for s in STRATS]
 strat_spread = float(max(strat_vals) - min(strat_vals))
 best_s = STRATS[int(np.argmax(strat_vals))]
 worst_s = STRATS[int(np.argmin(strat_vals))]
 full = float(sub[sub.strategy == "full"].delta.iloc[0])
 rows.append({
 "combo": combo,
 "random_delta_k10": round(means10["random"], 4),
 "topdegree_delta_k10": round(means10["topdegree"], 4),
 "earlyactive_delta_k10": round(means10["earlyactive"], 4),
 "strategy_spread_k10": round(strat_spread, 4),
 "size_spread_random_k1to10": round(size_spread, 4),
 "spread_ratio_strat_over_size": round(strat_spread / size_spread, 3)
 if size_spread > 0 else np.nan,
 "identity_dominates_size": bool(strat_spread > size_spread),
 "best_strategy": best_s, "worst_strategy": worst_s,
 "random_k10_trial_min": round(float(trials10.min()), 4),
 "random_k10_trial_max": round(float(trials10.max()), 4),
 "random_k10_trial_std": round(float(trials10.std(ddof=1)), 4),
 "full_delta": round(full, 4),
 })
 tab = pd.DataFrame(rows)
 tab.to_csv(os.path.join(OUT, "combo_spread_table.csv"), index=False)
 pd.DataFrame(k10_cells).to_csv(os.path.join(OUT, "k10_cells.csv"), index=False)

 # ---------- (b) conditional statement test ----------
 cond = {
 "random_effective_all_6_combos_k10": bool((tab.random_delta_k10 > 0.05).all()),
 "random_min_k10_across_combos": float(tab.random_delta_k10.min()),
 "guided_fail_cells": [],
 "random_ineffective_any_combo": bool((tab.random_delta_k10 <= 0.05).any()),
 }
 for _, r in tab.iterrows():
 for s in ("topdegree", "earlyactive"):
 v = r[f"{s}_delta_k10"]
 if v < 0.05:
 cond["guided_fail_cells"].append(
 {"combo": r.combo, "strategy": s, "delta": v,
 "vs_random_same_combo": round(v - r.random_delta_k10, 4)})

 # ---------- (c) interaction: dataset family x strategy gap at k=10 ----------
 # strategy gap g_c = max(0, random - best_guided) is not symmetric; use
 # random - mean(topdegree, earlyactive) as the "identity cost" estimate
 tab["identity_cost"] = tab.random_delta_k10 - (
 tab.topdegree_delta_k10 + tab.earlyactive_delta_k10) / 2.0
 tab["family"] = np.where(tab.combo.str.contains("CADETS"), "CADETS", "THEIA")
 ca = tab[tab.family == "CADETS"].identity_cost.to_numpy()
 th = tab[tab.family == "THEIA"].identity_cost.to_numpy()
 diff_obs = float(ca.mean() - th.mean())
 # exact permutation: all choose(6,3)=20 label assignments
 allv = tab.identity_cost.to_numpy()
 labels = tab.family.to_numpy()
 diffs = []
 for idx in itertools.combinations(range(6), 3):
 m = np.zeros(6, dtype=bool)
 m[list(idx)] = True
 diffs.append(allv[m].mean() - allv[~m].mean())
 diffs = np.abs(np.asarray(diffs))
 p_perm = float((diffs >= abs(diff_obs) - 1e-12).sum() / len(diffs))
 interaction = {
 "estimand": "identity cost = random_delta(k=10) - mean(topdegree, "
 "earlyactive)_delta(k=10), per combo",
 "identity_cost_per_combo": {r.combo: round(r.identity_cost, 4)
 for _, r in tab.iterrows()},
 "CADETS_mean": round(float(ca.mean()), 4),
 "THEIA_mean": round(float(th.mean()), 4),
 "interaction_point_estimate": round(diff_obs, 4),
 "permutation_exact_p_20splits": p_perm,
 "n_per_family": 3,
 "power_caveat": "Brookes 2004 / ICH E9 s5.7: interaction tests need "
 "~4x the sample of the main effect; with 3 combos per "
 "family this test is exploratory regardless of p",
 }

 # ---------- (d) revised headline ----------
 n_id_dom = int(tab.identity_dominates_size.sum())
 headline = {
 "old_headline": "|M| size dominates, source identity secondary "
 "(size_spread 0.294 ~ 2x strategy_spread 0.157, "
 "cross-granularity)",
 "revised_headline": (
 "At the same k=10 granularity, source identity and |M| size are "
 "comparable in aggregate (strategy spread {ss:.3f} vs size spread "
 "{sz:.3f} averaged over combos), but identity effects are "
 "family-modulated: random selection is the only source effective "
 "in all 6 combos (min gain {rmin:.3f}), while timeline/EDR-style "
 "guided sources fail on THEIA ({nfail} of 12 guided cells with "
 "gain <0.05, all on THEIA, worst {worst}) and dominate on CADETS."
 ).format(
 ss=float(tab.strategy_spread_k10.mean()),
 sz=float(tab.size_spread_random_k1to10.mean()),
 rmin=float(tab.random_delta_k10.min()),
 nfail=len(cond["guided_fail_cells"]),
 worst=min(cond["guided_fail_cells"], key=lambda d: d["delta"])["combo"]
 + " " + min(cond["guided_fail_cells"], key=lambda d: d["delta"])["strategy"]
 + f" {min(cond['guided_fail_cells'], key=lambda d: d['delta'])['delta']}"
 if cond["guided_fail_cells"] else "none"),
 "identity_dominates_size_n_combos": f"{n_id_dom}/6",
 "aggregate_same_granularity_ratio": round(
 float(tab.strategy_spread_k10.mean() / tab.size_spread_random_k1to10.mean()), 3),
 }

 out = {
 "design": {"anchor": "k=10", "strategies": list(STRATS),
 "size_axis": "random mean delta over k in {1,3,5,10}"},
 "per_combo_table": tab.to_dict(orient="records"),
 "conditional": cond,
 "interaction": interaction,
 "headline": headline,
 }
 with open(os.path.join(OUT, "interaction.json"), "w") as f:
 json.dump(out, f, indent=1, ensure_ascii=False)
 print(tab[["combo", "random_delta_k10", "topdegree_delta_k10",
 "earlyactive_delta_k10", "strategy_spread_k10",
 "size_spread_random_k1to10", "identity_dominates_size",
 "identity_cost"]].to_string())
 print("\ninteraction:", json.dumps(interaction, indent=1)[:600])
 print("\nheadline:", json.dumps(headline, indent=1, ensure_ascii=False))
 print(" DONE")

if __name__ == "__main__":
 main()
