#!/usr/bin/env python3
""": TRUE-dilution re-run of / (fixes R2A-C1/R2B-C1 no-op Critical).

The archived noise_alpha.py:120 / fp_noise.py:116 filtered
long_df to .isin(all_mal), so FALSE endpoints never entered the ICE table
and CI(e) never saw them: the archived "bit-identical ADP under 10-50%
false-endpoint dilution" was a constructive no-op, not a robustness result
(both R1 and R2 review panels reproduced this independently).

This script implements the protocol the paper Eq.(3) actually declares:
 CI(e) = (1/|M∩N(e)|) * sum_{v in M∩N(e)} ICE(e,v),
with the ICE table computed over ALL nodes (true + benign), so false
endpoints genuinely enter the mean denominator and the support set.
Implementation spec fixed per R2B-m3: full-node ICE table + explicit
aggregation caliber (mean over M∩N(e), then sum CI(e) onto both endpoints
of e). Arms identical to (ca sigmoid / adj_half hard mask / adj_sigma)
plus support_size; noise forms: uniform benign pool () and top-100
high-scoring FP pool (). Seeds identical to the archived runs
(seed*10000+int(p*100)) for comparability.

Sanity anchors written to the json:
 - p0 must be bit-identical to the archived e24 p0 (at p=0, M=full_M,
 the full-node table restricted to full_M equals the old mal-only table);
 - per-trial diagnostics: support_size, n_fake_with_pos_ci, support_gain.

Output: results/reference_contamination/e24c_<model>_<dataset>_ep<N>.json + summary
Usage: python3 reference_contamination_dilution.py ALL
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

WS = os.environ.get("CAUSAL_WORKSPACE", ".")
TRAIN = os.path.join(WS, "artifacts/training/training")
OUT = os.path.join(WS, "results/reference_contamination")
os.makedirs(OUT, exist_ok=True)

HASH = {
 ("orthrus", "CADETS_E3"): "266a87dde3fae4c72f17282bc10c4ad1cecef69c55f8e47796d65fedc1601de9",
 ("velox", "CADETS_E3"): "ea29a6902c6651cf2308eaa76e0523772454ce42f4c50e77cd99c932e211ce4a",
 ("kairos", "CADETS_E3"): "3128a7535d8a58b609c735f7745ebcede2f528d0ea98afd93c5923fd7f92b191",
 ("orthrus", "THEIA_E3"): "7f55817b33b4c2a327cd1e669799954fdc64278dd6cfe2ae444af4c6d9773f76",
 ("velox", "THEIA_E3"): "7a57b7274859f28b15dabf945270357ac668ed7a40b0c2ba6c2517c57ec9977c",
 ("kairos", "THEIA_E3"): "1a43f8e8253beb62033f23193471422a0a96647e4113d1b6e1faae4f91697505",
}
GT_BASE = os.path.join(WS, "code/PIDSMaker/Ground_Truth/orthrus")
DATASET_ATTACKS = {
 "CADETS_E3": ["E3-CADETS/node_Nginx_Backdoor_06.csv", "E3-CADETS/node_Nginx_Backdoor_12.csv",
 "E3-CADETS/node_Nginx_Backdoor_13.csv"],
 "THEIA_E3": ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
 "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv"],
}
ALPHA = 1.0
NOISE_P = [0.10, 0.30, 0.50]
N_SEED = 5
ALPHA_SCAN = [0.5, 1.0, 2.0, 5.0]
COMBOS = [("kairos", "CADETS_E3", 13), ("kairos", "THEIA_E3", 19),
 ("velox", "CADETS_E3", 13), ("velox", "THEIA_E3", 19),
 ("orthrus", "CADETS_E3", 13), ("orthrus", "THEIA_E3", 19)]

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

def load_gt(dataset):
 attack_to_mal, node_to_attacks = {}, {}
 for i, f in enumerate(DATASET_ATTACKS[dataset]):
 df = pd.read_csv(os.path.join(GT_BASE, f), header=None, names=["uuid", "label", "index_id"])
 ids = set(df["index_id"].astype(int))
 attack_to_mal[f"attack_{i}"] = ids
 for n in ids:
 node_to_attacks.setdefault(n, set()).add(f"attack_{i}")
 return attack_to_mal, node_to_attacks

def full_node_ice(df):
 """ICE over ALL nodes (vectorized). Same semantics as the archived
 per-node apply(nlargest(2)) but computed for every node once."""
 n = len(df)
 src = pd.DataFrame({"node": df["srcnode"].values, "loss": df["loss"].values,
 "edge_idx": np.arange(n)})
 dst = pd.DataFrame({"node": df["dstnode"].values, "loss": df["loss"].values,
 "edge_idx": np.arange(n)})
 long_df = pd.concat([src, dst], ignore_index=True)
 long_sorted = long_df.sort_values(["node", "loss"], ascending=[True, False],
 kind="mergesort")
 top2 = long_sorted.groupby("node", sort=False).head(2)
 g = top2.groupby("node")["loss"]
 node_max = g.max()
 node_second = g.min()
 cnt = g.size()
 node_second = node_second.where(cnt >= 2, 0.0) # single-edge node -> 0.0
 long_df = long_df.merge(node_max.rename("n_max"), left_on="node", right_index=True)
 long_df = long_df.merge(node_second.rename("n_second"), left_on="node", right_index=True)
 long_df["is_max"] = long_df["loss"] == long_df["n_max"]
 cm = long_df.groupby("node")["is_max"].transform("sum")
 uniq = long_df["is_max"] & (cm == 1)
 long_df["ice"] = np.where(uniq, (long_df["n_max"] - long_df["n_second"]).abs(), 0.0)
 return long_df[["edge_idx", "node", "ice"]]

def run_combo(model, dataset, epoch):
 t0 = time.time()
 h = HASH[(model, dataset)]
 attack_to_mal, node_to_attacks = load_gt(dataset)
 all_mal = set().union(*attack_to_mal.values())
 losses = os.path.join(TRAIN, h, dataset, "edge_losses", "test", f"model_epoch_{epoch}")
 files = sorted(glob.glob(os.path.join(losses, "*.csv")))
 with ThreadPoolExecutor(max_workers=48) as ex:
 dfs = list(ex.map(pd.read_csv, files))
 df = pd.concat(dfs, ignore_index=True)
 df["srcnode"] = df["srcnode"].astype("int64")
 df["dstnode"] = df["dstnode"].astype("int64")
 print(f"{model}_{dataset}_ep{epoch}: {len(df)} edges, {len(all_mal)} mal", flush=True)

 g = pd.concat([df.groupby("srcnode")["loss"].max(), df.groupby("dstnode")["loss"].max()])
 nmax = g.groupby(level=0).max()
 nodes = list(nmax.index)
 node_set = set(nodes)
 labels = [1 if n in all_mal else 0 for n in nodes]
 s_arr = df["srcnode"].values
 d_arr = df["dstnode"].values

 ice_tbl = full_node_ice(df) # THE fix: all nodes, not just all_mal
 print(f" full-node ICE table: {ice_tbl.shape[0]} rows, "
 f"{ice_tbl['node'].nunique()} nodes ({time.time()-t0:.1f}s)", flush=True)

 benign_pool = sorted(node_set - all_mal)
 benign_scores = [(nmax[n], n) for n in nodes if n not in all_mal]
 benign_scores.sort(reverse=True)
 fp_pool = [n for _, n in benign_scores[:100]]

 def ci_sum_for(M):
 sub = ice_tbl[ice_tbl["node"].isin(M)]
 if sub.empty:
 return {}
 ci_edge = sub.groupby("edge_idx")["ice"].mean() # Eq.(3): 1/|M∩N(e)|
 ci_s = {}
 for ei, v in ci_edge.items():
 ci_s[s_arr[ei]] = ci_s.get(s_arr[ei], 0.0) + v
 ci_s[d_arr[ei]] = ci_s.get(d_arr[ei], 0.0) + v
 return ci_s

 def arms_for(M, alpha=ALPHA):
 ci_s = ci_sum_for(M)
 ci_vec = np.array([ci_s.get(nn, 0.0) for nn in nodes])
 support = ci_vec > 0
 ca = nmax.values * (1.0 / (1.0 + np.exp(-alpha * ci_vec)))
 return {
 "ca": round(compute_adp(ca, nodes, node_to_attacks, labels), 4),
 "adj_half": round(compute_adp(nmax.values * np.where(support, 1.0, 0.5),
 nodes, node_to_attacks, labels), 4),
 "adj_sigma": round(compute_adp(nmax.values * np.where(support, 1.0 / (1.0 + np.exp(-alpha)), 0.5),
 nodes, node_to_attacks, labels), 4),
 "support_size": int(support.sum()),
 "n_fake_with_pos_ci": int(sum(1 for f in (M - all_mal) if ci_s.get(f, 0.0) > 0.0)),
 }

 import random
 full_M = sorted(all_mal)
 results = {"model": model, "dataset": dataset, "epoch": epoch,
 "ice_semantics": "corrected margin, FULL-NODE table (true dilution, fixes R2A-C1/R2B-C1)",
 "aggregation_caliber": "CI(e)=mean over M∩N(e) per paper Eq.(3); sum CI(e) onto both endpoints",
 "full_M": len(full_M), "noise_forms": ["uniform_benign", "top100_fp"]}

 # clean reference (p=0) -- must be bit-identical to archived e24 p0
 results["p0"] = arms_for(set(full_M))

 # alpha scan at p=0
 results["alpha_scan"] = {str(a): arms_for(set(full_M), alpha=a)["ca"] for a in ALPHA_SCAN}

 def noise_block(pool, pool_name):
 noise = {}
 for p in NOISE_P:
 n_fake = max(1, round(p * len(full_M)))
 trials = []
 for seed in range(N_SEED):
 random.seed(seed * 10000 + int(p * 100))
 fake = set(random.sample(pool, min(n_fake, len(pool))))
 M = set(full_M) | fake
 arms = arms_for(M)
 arms["n_fake"] = len(fake)
 arms["support_gain"] = arms["support_size"] - results["p0"]["support_size"]
 trials.append(arms)
 agg = {k + "_mean": round(float(np.mean([t[k] for t in trials])), 4)
 for k in ["ca", "adj_half", "adj_sigma", "support_size", "n_fake_with_pos_ci"]}
 agg["ca_max_abs_delta_vs_p0"] = round(max(abs(t["ca"] - results["p0"]["ca"]) for t in trials), 4)
 agg["adj_half_max_abs_delta_vs_p0"] = round(max(abs(t["adj_half"] - results["p0"]["adj_half"]) for t in trials), 4)
 agg["trials"] = trials
 noise[f"p{int(p*100)}"] = agg
 return noise

 results["noise_uniform_benign"] = noise_block(benign_pool, "uniform_benign")
 results["noise_top100_fp"] = noise_block(fp_pool, "top100_fp")

 out = os.path.join(OUT, f"e24c_{model}_{dataset}_ep{epoch}.json")
 with open(out, "w") as f:
 json.dump(results, f, indent=1)
 p0 = results["p0"]
 nu, nf = results["noise_uniform_benign"], results["noise_top100_fp"]
 print(f"saved {out} ({time.time()-t0:.1f}s)", flush=True)
 print(f" p0: ca={p0['ca']} mask={p0['adj_half']} support={p0['support_size']}", flush=True)
 print(f" uniform p30: ca={nu['p30']['ca_mean']} (maxΔ {nu['p30']['ca_max_abs_delta_vs_p0']}) "
 f"mask={nu['p30']['adj_half_mean']} (maxΔ {nu['p30']['adj_half_max_abs_delta_vs_p0']}) "
 f"support_gain={nu['p30']['trials'][0]['support_gain']}", flush=True)
 print(f" top100fp p30: ca={nf['p30']['ca_mean']} (maxΔ {nf['p30']['ca_max_abs_delta_vs_p0']}) "
 f"mask={nf['p30']['adj_half_mean']} (maxΔ {nf['p30']['adj_half_max_abs_delta_vs_p0']}) "
 f"support_gain={nf['p30']['trials'][0]['support_gain']}", flush=True)
 return results

def main():
 if len(sys.argv) > 1 and sys.argv[1] == "ALL":
 allres = [run_combo(m, d, e) for m, d, e in COMBOS]
 with open(os.path.join(OUT, "summary.json"), "w") as f:
 json.dump({"combos": allres}, f, indent=1)
 print("summary saved", flush=True)
 else:
 model, dataset, epoch = sys.argv[1], sys.argv[2], int(sys.argv[3])
 run_combo(model, dataset, epoch)

if __name__ == "__main__":
 main()
