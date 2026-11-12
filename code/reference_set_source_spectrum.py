#!/usr/bin/env python3
""": M-source quality spectrum — does CA gain depend on HOW M was obtained?

Motivation (21A): CA assumes a reference set M of known malicious endpoints.
Realistic M sources differ in WHICH nodes they capture, not just how many:
 - TI/IOC matching -> effectively a random-ish subset of malicious nodes
 - EDR beacons -> attack-hub nodes (high-degree core) get beaconed first
 - forensic timeline-> earliest-active attack nodes (timeline head)

Design: per (model, dataset, epoch) representative checkpoint, construct M by
 strategy in {random, top-degree, early-active} x |M| in {1,3,5,10} (+full M
 upper bound), 5 sampling trials for stochastic strategies, CORRECTED ICE
 (sparse precomputed table, identical to stage3 fix 2026-10-11).
Outcome: CA ADP and delta vs baseline. Hypothesis under test: gain is more
sensitive to |M| than to source identity (supports 'any of the three realistic
sources works' narrative); violations reported honestly.

Output: results/reference_set_sensitivity/msource_grid.csv + msource_summary.json

Usage: python3 reference_set_source_spectrum.py
"""
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

WS = os.environ.get("CAUSAL_WORKSPACE", ".")
TRAIN = os.path.join(WS, "artifacts/training/training")
OUT = os.path.join(WS, "results/reference_set_sensitivity")
os.makedirs(OUT, exist_ok=True)

COMBOS = [ # (model, dataset, epoch, hash)
 ("orthrus", "CADETS_E3", 13, "266a87dde3fae4c72f17282bc10c4ad1cecef69c55f8e47796d65fedc1601de9"),
 ("velox", "CADETS_E3", 13, "ea29a6902c6651cf2308eaa76e0523772454ce42f4c50e77cd99c932e211ce4a"),
 ("kairos", "CADETS_E3", 13, "3128a7535d8a58b609c735f7745ebcede2f528d0ea98afd93c5923fd7f92b191"),
 ("orthrus", "THEIA_E3", 19, "7f55817b33b4c2a327cd1e669799954fdc64278dd6cfe2ae444af4c6d9773f76"),
 ("velox", "THEIA_E3", 19, "7a57b7274859f28b15dabf945270357ac668ed7a40b0c2ba6c2517c57ec9977c"),
 ("kairos", "THEIA_E3", 19, "1a43f8e8253beb62033f23193471422a0a96647e4113d1b6e1faae4f91697505"),
]
GT_BASE = os.path.join(WS, "code/PIDSMaker/Ground_Truth/orthrus")
GT_FILES = {
 "CADETS_E3": ["E3-CADETS/node_Nginx_Backdoor_06.csv", "E3-CADETS/node_Nginx_Backdoor_12.csv",
 "E3-CADETS/node_Nginx_Backdoor_13.csv"],
 "THEIA_E3": ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
 "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv"],
}
SIZES = [1, 3, 5, 10]
STRATEGIES = ["random", "topdegree", "earlyactive"]
N_TRIALS = 5
ALPHA = 1.0
RNG_SEED = 20261011

def compute_adp(scores, nodes, node2attacks, labels):
 scores = np.asarray(scores, float)
 labels = np.asarray(labels, int)
 order = np.argsort(scores)[::-1]
 total_attacks = len({a for atks in node2attacks.values() for a in atks})
 if total_attacks == 0:
 return 0.0
 detected, tp, fp = set(), 0, 0
 prec_to_max = {}
 for i in order:
 if labels[i] == 1:
 tp += 1
 detected.update(node2attacks.get(nodes[i], ()))
 else:
 fp += 1
 p = tp / (tp + fp) if (tp + fp) else 0.0
 d = len(detected) / total_attacks * 100
 prec_to_max[p] = max(prec_to_max.get(p, 0.0), d)
 sp = sorted(prec_to_max)
 return float(np.trapz([prec_to_max[p] for p in sp], sp) / 100.0)

def run_combo(model, dataset, epoch, h):
 tag = f"{model}_{dataset}_ep{epoch}"
 attack_to_mal, node_to_attacks = {}, {}
 for i, f in enumerate(GT_FILES[dataset]):
 g = pd.read_csv(os.path.join(GT_BASE, f), header=None, names=["uuid", "label", "index_id"])
 ids = set(g["index_id"].astype(int))
 attack_to_mal[f"attack_{i}"] = ids
 for nn in ids:
 node_to_attacks.setdefault(nn, set()).add(f"attack_{i}")
 all_mal = set().union(*attack_to_mal.values())

 losses = os.path.join(TRAIN, h, dataset, "edge_losses/test", f"model_epoch_{epoch}")
 files = sorted(glob.glob(os.path.join(losses, "*.csv")))
 with ThreadPoolExecutor(max_workers=48) as ex:
 dfs = list(ex.map(pd.read_csv, files))
 df = pd.concat(dfs, ignore_index=True)
 df["srcnode"] = df["srcnode"].astype("int64")
 df["dstnode"] = df["dstnode"].astype("int64")
 df["time"] = df["time"].astype("int64")

 g = pd.concat([df.groupby("srcnode")["loss"].max(), df.groupby("dstnode")["loss"].max()])
 nmax = g.groupby(level=0).max()
 nodes = list(nmax.index)
 labels = [1 if n in all_mal else 0 for n in nodes]
 baseline_adp = compute_adp(nmax.values, nodes, node_to_attacks, labels)

 # sparse corrected ICE over full malicious pool
 n = len(df)
 src = pd.DataFrame({"node": df["srcnode"].values, "loss": df["loss"].values, "edge_idx": np.arange(n)})
 dst = pd.DataFrame({"node": df["dstnode"].values, "loss": df["loss"].values, "edge_idx": np.arange(n)})
 long_df = pd.concat([src, dst], ignore_index=True)
 mal_long = long_df[long_df["node"].isin(all_mal)]
 grp = mal_long.groupby("node")["loss"]
 node_max = grp.max()
 node_second = grp.apply(lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else 0.0)
 mal_long = mal_long.merge(node_max.rename("n_max"), left_on="node", right_index=True)
 mal_long = mal_long.merge(node_second.rename("n_second"), left_on="node", right_index=True)
 mal_long["is_max"] = mal_long["loss"] == mal_long["n_max"]
 cm = mal_long.groupby("node")["is_max"].transform("sum")
 uniq = mal_long["is_max"] & (cm == 1)
 mal_long["ice"] = np.where(uniq, (mal_long["n_max"] - mal_long["n_second"]).abs(), 0.0)
 ice_tbl = mal_long[["edge_idx", "node", "ice"]]

 s_arr = df["srcnode"].values
 d_arr = df["dstnode"].values
 t_arr = df["time"].values

 def ca_adp_for(M):
 sub = ice_tbl[ice_tbl["node"].isin(set(M))]
 ci_s = {}
 if not sub.empty:
 for ei, v in sub.groupby("edge_idx")["ice"].mean().items():
 ci_s[s_arr[ei]] = ci_s.get(s_arr[ei], 0.0) + v
 ci_s[d_arr[ei]] = ci_s.get(d_arr[ei], 0.0) + v
 ca = nmax.values * (1.0 / (1.0 + np.exp(-ALPHA * np.array([ci_s.get(nn, 0.0) for nn in nodes]))))
 return compute_adp(ca, nodes, node_to_attacks, labels)

 # per-malicious-node features for deterministic strategies
 mal_list = sorted(all_mal)
 deg = mal_long.groupby("node").size()
 first_t = mal_long.groupby("node")["edge_idx"].apply(
 lambda s: int(t_arr[s.to_numpy()].min()))
 # NOTE edge_idx in mal_long refers to positions in the ORIGINAL wide df
 by_degree = sorted(mal_list, key=lambda nn: -deg.get(nn, 0))
 by_early = sorted(mal_list, key=lambda nn: first_t.get(nn, np.iinfo(np.int64).max))

 rows = []
 rng = np.random.default_rng(RNG_SEED)
 full_adp = ca_adp_for(mal_list)
 rows.append({"combo": tag, "strategy": "full", "k": len(mal_list), "trial": 0,
 "adp": round(full_adp, 4), "delta": round(full_adp - baseline_adp, 4)})
 for k in SIZES:
 k_eff = min(k, len(mal_list))
 for strat, cand in (("random", None), ("topdegree", by_degree), ("earlyactive", by_early)):
 trials = N_TRIALS if strat == "random" else 1
 for tr in range(trials):
 if strat == "random":
 M = list(rng.choice(mal_list, size=k_eff, replace=False))
 else:
 M = cand[:k_eff]
 adp = ca_adp_for(M)
 rows.append({"combo": tag, "strategy": strat, "k": k_eff, "trial": tr,
 "adp": round(adp, 4), "delta": round(adp - baseline_adp, 4)})
 print(f"{tag}: baseline={baseline_adp:.4f} full={full_adp:.4f} done", flush=True)
 return baseline_adp, rows

def main():
 all_rows = []
 bases = {}
 for model, dataset, epoch, h in COMBOS:
 b, rows = run_combo(model, dataset, epoch, h)
 bases[f"{model}_{dataset}_ep{epoch}"] = round(b, 4)
 all_rows.extend(rows)
 grid = pd.DataFrame(all_rows)
 grid["baseline_adp"] = grid["combo"].map(bases)
 grid.to_csv(os.path.join(OUT, "msource_grid.csv"), index=False)

 # summary: variance decomposition — how much does strategy identity matter
 # vs |M|, measured by spread of mean delta at each (combo, k)
 piv = grid[grid["strategy"] != "full"].groupby(["combo", "k", "strategy"])["delta"].mean().reset_index()
 spread_strategy = piv.groupby(["combo", "k"])["delta"].agg(lambda s: s.max() - s.min())
 spread_size = piv.groupby(["combo", "strategy"])["delta"].agg(lambda s: s.max() - s.min())
 summary = {
 "baseline_adps": bases,
 "mean_strategy_spread_within_combo_k": round(float(spread_strategy.mean()), 4),
 "mean_size_spread_within_combo_strategy": round(float(spread_size.mean()), 4),
 "interpretation_hint": "size_spread >> strategy_spread supports '|M| matters more than source identity'",
 "per_combo": {c: {"baseline": bases[c],
 "full_delta": float(grid[(grid.combo == c) & (grid.strategy == "full")]["delta"].iloc[0]),
 "best_k10_strategy": piv[(piv.combo == c) & (piv.k >= 10)]
 .groupby("strategy")["delta"].mean().round(4).to_dict()}
 for c in bases},
 }
 with open(os.path.join(OUT, "msource_summary.json"), "w") as f:
 json.dump(summary, f, indent=1)
 print(json.dumps(summary["per_combo"], indent=1), flush=True)
 print("saved", os.path.join(OUT, "msource_grid.csv"), flush=True)

if __name__ == "__main__":
 main()
