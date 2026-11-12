#!/usr/bin/env python3
""": adjacency-gate ablation (review-panel P0-2).

Devil's-advocate challenge: CI>0 iff an edge is adjacent to M, so the CA gate's
gain may come entirely from label-proximity injection (which nodes get boosted)
rather than intervention semantics (how much CI they carry). Ablation arms on
the SAME support set S = {v : sum_ci(v) > 0}:

 baseline : node max score s(v)
 ca : s(v) * sigma(alpha * sum_ci(v)) [paper, alpha=1]
 adj_sigma : s(v) * sigma(alpha * 1[sum_ci(v) > 0]) [binary adjacency gate]
 adj_half : s(v) * (1.0 if sum_ci(v)>0 else 0.5) [devil's verbatim mask]
 random_gate: s(v) * sigma(alpha * 1[v in R]), R random, |R|=|S|, 5 seeds

Settings: full M (Table I protocol) and |M|=10 random (seeds identical to
exp_b_fast trial*100+size so the ca arm must reproduce expb numbers
bit-for-bit = verification anchor).

Output: results/gate_form_ablation/e23_<model>_<dataset>_ep<N>.json per combo
 + adjacency_gate_summary.json

Usage: python3 gate_form_ablation.py <model> <dataset> <epoch>
 python3 gate_form_ablation.py ALL
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
OUT = os.path.join(WS, "results/gate_form_ablation")
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

 n = len(df)
 src = pd.DataFrame({"node": s_arr, "loss": df["loss"].values, "edge_idx": np.arange(n)})
 dst = pd.DataFrame({"node": d_arr, "loss": df["loss"].values, "edge_idx": np.arange(n)})
 long_df = pd.concat([src, dst], ignore_index=True)
 long_df = long_df[long_df["node"].isin(all_mal)]
 grp = long_df.groupby("node")["loss"]
 node_max = grp.max()
 node_second = grp.apply(lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else 0.0)
 long_df = long_df.merge(node_max.rename("n_max"), left_on="node", right_index=True)
 long_df = long_df.merge(node_second.rename("n_second"), left_on="node", right_index=True)
 long_df["is_max"] = long_df["loss"] == long_df["n_max"]
 cm = long_df.groupby("node")["is_max"].transform("sum")
 uniq = long_df["is_max"] & (cm == 1)
 long_df["ice"] = np.where(uniq, (long_df["n_max"] - long_df["n_second"]).abs(), 0.0)
 ice_tbl = long_df[["edge_idx", "node", "ice"]]

 def ci_sum_for(M):
 sub = ice_tbl[ice_tbl["node"].isin(M)]
 if sub.empty:
 return {}
 ci_edge = sub.groupby("edge_idx")["ice"].mean()
 ci_s = {}
 for ei, v in ci_edge.items():
 ci_s[s_arr[ei]] = ci_s.get(s_arr[ei], 0.0) + v
 ci_s[d_arr[ei]] = ci_s.get(d_arr[ei], 0.0) + v
 return ci_s

 import random

 def arm_adps_for(M, tag):
 ci_s = ci_sum_for(M)
 ci_vec = np.array([ci_s.get(nn, 0.0) for nn in nodes])
 support = ci_vec > 0
 sup_idx = np.where(support)[0]
 base = compute_adp(nmax.values, nodes, node_to_attacks, labels)
 ca = nmax.values * (1.0 / (1.0 + np.exp(-ALPHA * ci_vec)))
 res = {"setting": tag, "|M|": len(M), "support_size": int(support.sum()),
 "support_mal_count": int(sum(labels[i] for i in sup_idx)),
 "baseline": round(base, 4),
 "ca": round(compute_adp(ca, nodes, node_to_attacks, labels), 4),
 "adj_sigma": round(compute_adp(nmax.values * np.where(support, 1.0 / (1.0 + np.exp(-ALPHA)), 0.5),
 nodes, node_to_attacks, labels), 4),
 "adj_half": round(compute_adp(nmax.values * np.where(support, 1.0, 0.5),
 nodes, node_to_attacks, labels), 4)}
 rg = []
 for rseed in range(5):
 _tag_hash = int.from_bytes(__import__('hashlib').md5(str(tag).encode()).digest()[:4], 'little')
 random.seed(rseed * 1000 + _tag_hash % 9973 + len(M)) # stable hash (R2A-m2 fix; was hash(tag), cross-process unstable)
 R = set(random.sample(sorted(node_set), int(support.sum())))
 rg.append(round(compute_adp(nmax.values * np.array([1.0 / (1.0 + np.exp(-ALPHA * (1.0 if nn in R else 0.0)))
 for nn in nodes]),
 nodes, node_to_attacks, labels), 4))
 res["random_gate_mean"] = round(float(np.mean(rg)), 4)
 res["random_gate_trials"] = rg
 return res

 results = {"model": model, "dataset": dataset, "epoch": epoch,
 "ice_semantics": "corrected margin (2026-10-11)", "alpha": ALPHA}

 # Setting A: full M (Table I protocol)
 full_M = sorted(all_mal)
 results["full_M"] = arm_adps_for(set(full_M), "full_M")

 # Setting B: |M|=10, seeds identical to exp_b_fast (trial*100+size)
 size = 10
 per_trial = []
 for trial in range(5):
 random.seed(trial * 100 + size)
 M = set(random.sample(full_M, size))
 per_trial.append(arm_adps_for(M, f"random10_trial{trial}"))
 agg = {}
 for k in ["baseline", "ca", "adj_sigma", "adj_half", "random_gate_mean", "support_size",
 "support_mal_count"]:
 vals = [t[k] for t in per_trial]
 agg[k + "_mean"] = round(float(np.mean(vals)), 4)
 agg[k + "_std"] = round(float(np.std(vals)), 4)
 results["M10"] = {"trials": per_trial, "aggregate": agg}

 out = os.path.join(OUT, f"e23_{model}_{dataset}_ep{epoch}.json")
 with open(out, "w") as f:
 json.dump(results, f, indent=1)
 print(f"saved {out} ({time.time()-t0:.1f}s)", flush=True)
 fm = results["full_M"]
 print(f" full_M: base={fm['baseline']} ca={fm['ca']} adj_sigma={fm['adj_sigma']} "
 f"adj_half={fm['adj_half']} rand={fm['random_gate_mean']} "
 f"(support={fm['support_size']}, mal_in_support={fm['support_mal_count']})", flush=True)
 return results

def main():
 if len(sys.argv) > 1 and sys.argv[1] == "ALL":
 allres = [run_combo(m, d, e) for m, d, e in COMBOS]
 with open(os.path.join(OUT, "adjacency_gate_summary.json"), "w") as f:
 json.dump({"combos": allres}, f, indent=1)
 print("summary saved", flush=True)
 else:
 model, dataset, epoch = sys.argv[1], sys.argv[2], int(sys.argv[3])
 run_combo(model, dataset, epoch)

if __name__ == "__main__":
 main()
