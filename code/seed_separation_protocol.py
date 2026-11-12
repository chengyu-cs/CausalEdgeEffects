#!/usr/bin/env python3
""": temporal-causal seed/eval separation, PAPER version with CORRECTED ICE.

Protocol (labels-only-from-past, strict seed/eval temporal separation):
 - t_cut = attack_start + d minutes (d in DELAYS)
 - seeds M(t_cut) = malicious nodes of attack a active in [tw_start, t_cut), sample k
 - CI computed ONLY from past edges (t < t_cut) + seed M (CORRECTED closed-form ICE:
 unique-max margin semantics = stage3_causal_pipeline.py fix 2026-10-11)
 - rerank future-edge node scores (fut_max); eval set = future-active nodes minus M
 - CA = fut_max * sigmoid(alpha * ci_sum_past)

Grid: delays {2,5,10,20} x |M| {1,3,5} x 20 seed-sampling repeats, per attack.
Controls: oracle (all pool seeds, still time-separated), random scores (10 draws),
cross-attack seeds (5 min after source attack ends).
Self-check: full-GT full-period CI must reproduce (corrected) ca_gt_adp.

Checkpoints (paper representative pair): kairos THEIA_E3 ep19 (weak) +
velox CADETS_E3 ep13 (strong). Hash identities per the run-hash mapping.

Output: results/seed_separation/grid_<tag>.csv + e12_<tag>.json

Usage: python3 seed_separation_protocol.py <model> <dataset> <epoch>
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

WS = os.environ.get("CAUSAL_WORKSPACE", ".")
TRAIN = os.path.join(WS, "artifacts/training/training")
GT_BASE = os.path.join(WS, "code/PIDSMaker/Ground_Truth/orthrus")
OUT = os.path.join(WS, "results/seed_separation")
os.makedirs(OUT, exist_ok=True)

HASH = {
 ("kairos", "THEIA_E3"): "1a43f8e8253beb62033f23193471422a0a96647e4113d1b6e1faae4f91697505",
 ("velox", "CADETS_E3"): "ea29a6902c6651cf2308eaa76e0523772454ce42f4c50e77cd99c932e211ce4a",
 ("kairos", "CADETS_E3"): "3128a7535d8a58b609c735f7745ebcede2f528d0ea98afd93c5923fd7f92b191",
 ("velox", "THEIA_E3"): "7a57b7274859f28b15dabf945270357ac668ed7a40b0c2ba6c2517c57ec9977c",
 ("orthrus", "CADETS_E3"): "266a87dde3fae4c72f17282bc10c4ad1cecef69c55f8e47796d65fedc1601de9",
 ("orthrus", "THEIA_E3"): "7f55817b33b4c2a327cd1e669799954fdc64278dd6cfe2ae444af4c6d9773f76",
}
DATASET_ATTACKS = {
 "CADETS_E3": {
 "Backdoor_06": ("E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"),
 "Backdoor_12": ("E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"),
 "Backdoor_13": ("E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"),
 },
 "THEIA_E3": {
 "Browser_Extension_Drakon": ("E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv", "2018-04-12 12:40:00", "2018-04-12 13:30:00"),
 "Firefox_Backdoor_Drakon": ("E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv", "2018-04-10 14:30:00", "2018-04-10 15:00:00"),
 },
}
DELAYS = [2, 5, 10, 20]
KS = [1, 3, 5]
R = 20
ALPHA = 1.0

def str_to_ns(s):
 import pytz
 from datetime import datetime
 tz = pytz.timezone("US/Eastern")
 return int(tz.localize(datetime.strptime(s, "%Y-%m-%d %H:%M:%S")).timestamp() * 1e9)

def load_gt(dataset):
 attack_to_mal, node_to_attacks, windows = {}, {}, {}
 for a, (f, s, e) in DATASET_ATTACKS[dataset].items():
 df = pd.read_csv(os.path.join(GT_BASE, f), header=None, names=["uuid", "label", "index_id"])
 ids = set(df["index_id"].astype(int))
 attack_to_mal[a] = ids
 windows[a] = (str_to_ns(s), str_to_ns(e))
 for n in ids:
 node_to_attacks.setdefault(n, set()).add(a)
 return attack_to_mal, node_to_attacks, windows

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
 n = nodes[i]
 if labels[i] == 1:
 tp += 1
 detected.update(node2attacks.get(n, ()))
 else:
 fp += 1
 p = tp / (tp + fp) if (tp + fp) else 0.0
 d = len(detected) / total_attacks * 100
 prec_to_max[p] = max(prec_to_max.get(p, 0.0), d)
 sp = sorted(prec_to_max)
 return float(np.trapz([prec_to_max[p] for p in sp], sp) / 100.0)

def p_at_k(scores, nodes, labels, k=10):
 order = np.argsort(np.asarray(scores, float))[::-1][:k]
 return float(np.mean(np.asarray(labels)[order]))

def node_max_loss(df):
 g = pd.concat([df.groupby("srcnode")["loss"].max(), df.groupby("dstnode")["loss"].max()])
 return g.groupby(level=0).max()

def ice_table_corrected(past_df, seed_set):
 """CORRECTED ICE (margin semantics), seed nodes only.

 Follows stage3_causal_pipeline.py fix 2026-10-11: per-group nlargest(2)
 instead of the buggy nth(1) reindex. Returns DataFrame[edge_idx, node, ice].
 """
 n = len(past_df)
 src = pd.DataFrame({"node": past_df["srcnode"].values, "loss": past_df["loss"].values,
 "edge_idx": np.arange(n)})
 dst = pd.DataFrame({"node": past_df["dstnode"].values, "loss": past_df["loss"].values,
 "edge_idx": np.arange(n)})
 long_df = pd.concat([src, dst], ignore_index=True)
 long_df = long_df[long_df["node"].isin(seed_set)]
 if long_df.empty:
 return long_df
 grp = long_df.groupby("node")["loss"]
 node_max = grp.max()
 # corrected second-max: nlargest(2).iloc[-1]; single-edge nodes keep 0
 node_second = grp.apply(lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else 0.0)
 long_df = long_df.merge(node_max.rename("n_max"), left_on="node", right_index=True)
 long_df = long_df.merge(node_second.rename("n_second"), left_on="node", right_index=True)
 long_df["is_max"] = long_df["loss"] == long_df["n_max"]
 cm = long_df.groupby("node")["is_max"].transform("sum")
 uniq = long_df["is_max"] & (cm == 1)
 long_df["ice"] = np.where(uniq, (long_df["n_max"] - long_df["n_second"]).abs(), 0.0)
 return long_df[["edge_idx", "node", "ice"]]

def ci_sum_from_edges(ci_edge, past_df):
 ci_s = {}
 if ci_edge is None or len(ci_edge) == 0:
 return ci_s
 s_arr = past_df["srcnode"].values
 d_arr = past_df["dstnode"].values
 for ei, v in ci_edge.items():
 ci_s[s_arr[ei]] = ci_s.get(s_arr[ei], 0.0) + v
 ci_s[d_arr[ei]] = ci_s.get(d_arr[ei], 0.0) + v
 return ci_s

def sigmoid(x):
 return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))

def ca_apply(fut_max, ci_sum, exclude):
 nodes = [n for n in fut_max.index if n not in exclude]
 base = fut_max.reindex(nodes).values
 ci = np.array([ci_sum.get(n, 0.0) for n in nodes])
 return base * sigmoid(ALPHA * ci), nodes

def eval_scores(scores, nodes, node2attacks, all_mal):
 labels = [1 if n in all_mal else 0 for n in nodes]
 adp = compute_adp(scores, nodes, node2attacks, labels)
 pk = p_at_k(scores, nodes, labels, 10)
 return {"adp": round(adp, 4), "p@10": round(pk, 4),
 "n_mal_eval": int(sum(labels))}

def load_epoch_df(epoch_dir):
 files = sorted(glob.glob(os.path.join(epoch_dir, "*.csv")))
 with ThreadPoolExecutor(max_workers=min(48, len(files))) as ex:
 dfs = list(ex.map(pd.read_csv, files))
 return pd.concat(dfs, ignore_index=True)

def main():
 model, dataset, epoch = sys.argv[1], sys.argv[2], int(sys.argv[3])
 h = HASH[(model, dataset)]
 tag = f"{model}_{dataset}_ep{epoch}"
 attack_to_mal, node_to_attacks, windows = load_gt(dataset)
 all_mal = set().union(*attack_to_mal.values())
 losses = os.path.join(TRAIN, h, dataset, "edge_losses", "test", f"model_epoch_{epoch}")
 print(f"=== {tag} ({h[:8]}) loading", flush=True)
 df = load_epoch_df(losses)
 df["srcnode"] = df["srcnode"].astype("int64")
 df["dstnode"] = df["dstnode"].astype("int64")
 df["time"] = df["time"].astype("int64")
 print(f"loaded {len(df)} edges", flush=True)

 out = {"hash": h[:8], "model": model, "dataset": dataset, "epoch": epoch, "n_edges": len(df)}

 # ---------- 0. self-check: full-GT full-period corrected CI ----------
 ice_all_full = ice_table_corrected(df, all_mal)
 ci_full = ci_sum_from_edges(
 ice_all_full.groupby("edge_idx")["ice"].mean() if not ice_all_full.empty else pd.Series(dtype=float), df)
 nmax_full = node_max_loss(df)
 bl_full = eval_scores(nmax_full.values, list(nmax_full.index), node_to_attacks, all_mal)
 ca_vals = nmax_full.values * sigmoid(ALPHA * np.array([ci_full.get(n, 0.0) for n in nmax_full.index]))
 ca_full = eval_scores(ca_vals, list(nmax_full.index), node_to_attacks, all_mal)
 out["selfcheck_fullgt_corrected"] = {"baseline": bl_full, "ca_gt": ca_full}
 print("selfcheck(corrected full-GT): baseline", bl_full["adp"], " ca_gt", ca_full["adp"], flush=True)

 # ---------- 1. temporal-causal grid ----------
 rows = []
 for a, mal_a in attack_to_mal.items():
 s_ns, e_ns = windows[a]
 for d in DELAYS:
 t_cut = s_ns + int(d * 60e9)
 if t_cut >= e_ns:
 rows.append({"attack": a, "delay_min": d, "note": "window_too_short"})
 continue
 past = df[df["time"] < t_cut]
 fut = df[df["time"] >= t_cut]
 if past.empty or fut.empty:
 rows.append({"attack": a, "delay_min": d, "note": "empty_split"})
 continue
 active = set(past["srcnode"]) | set(past["dstnode"])
 seeds_pool = sorted(mal_a & active)
 if not seeds_pool:
 rows.append({"attack": a, "delay_min": d, "note": "no_active_seed", "pool": 0})
 continue
 fut_max = node_max_loss(fut)
 ice_all = ice_table_corrected(past, set(seeds_pool))
 ci_edge_or = (ice_all.groupby("edge_idx")["ice"].mean()
 if not ice_all.empty else pd.Series(dtype=float))
 ci_sum_or = ci_sum_from_edges(ci_edge_or, past)
 # oracle: all pool seeds
 sc_or, nd_or = ca_apply(fut_max, ci_sum_or, exclude=set(seeds_pool))
 ev_or = eval_scores(sc_or, nd_or, node_to_attacks, all_mal)
 # baseline with same pool exclusion
 bl_nodes = [n for n in fut_max.index if n not in set(seeds_pool)]
 ev_bl = eval_scores(fut_max.reindex(bl_nodes).values, bl_nodes, node_to_attacks, all_mal)
 # random scores control
 rng_r = np.random.default_rng(1010)
 rnd = [eval_scores(rng_r.random(len(bl_nodes)), bl_nodes, node_to_attacks, all_mal)["adp"]
 for _ in range(10)]
 row = {"attack": a, "delay_min": d, "pool": len(seeds_pool),
 "n_past_edges": len(past), "n_fut_edges": len(fut),
 "baseline_adp": ev_bl["adp"], "random_adp_mean": round(float(np.mean(rnd)), 4),
 "oracle_adp": ev_or["adp"], "oracle_delta": round(ev_or["adp"] - ev_bl["adp"], 4)}
 rng = np.random.default_rng(20261008 + d * 100 + len(seeds_pool))
 for k in KS:
 adps, p10s = [], []
 for r in range(R):
 k_eff = min(k, len(seeds_pool))
 sample = set(rng.choice(seeds_pool, size=k_eff, replace=False))
 if not ice_all.empty:
 sub = ice_all[ice_all["node"].isin(sample)]
 ci_e = sub.groupby("edge_idx")["ice"].mean()
 ci_s = ci_sum_from_edges(ci_e, past)
 else:
 ci_s = {}
 sc, nd = ca_apply(fut_max, ci_s, exclude=sample)
 ev = eval_scores(sc, nd, node_to_attacks, all_mal)
 adps.append(ev["adp"]); p10s.append(ev["p@10"])
 row[f"k{k}_adp_mean"] = round(float(np.mean(adps)), 4)
 row[f"k{k}_adp_std"] = round(float(np.std(adps)), 4)
 row[f"k{k}_delta_mean"] = round(float(np.mean(adps)) - ev_bl["adp"], 4)
 row[f"k{k}_p10_mean"] = round(float(np.mean(p10s)), 4)
 row[f"k{k}_pos_rate"] = round(float(np.mean([x > ev_bl["adp"] + 1e-9 for x in adps])), 3)
 rows.append(row)
 print(f"{a} d={d} pool={len(seeds_pool)} bl={ev_bl['adp']} rnd={row['random_adp_mean']} "
 f"oracle={ev_or['adp']} k1={row['k1_adp_mean']} k3={row['k3_adp_mean']} k5={row['k5_adp_mean']}",
 flush=True)
 grid = pd.DataFrame(rows)
 grid.to_csv(os.path.join(OUT, f"grid_{tag}.csv"), index=False)

 # ---------- 2. cross-attack seeds ----------
 xres = {}
 for a, mal_a in attack_to_mal.items():
 _, e_ns = windows[a]
 t_cut = e_ns + int(5 * 60e9)
 past = df[df["time"] < t_cut]
 fut = df[df["time"] >= t_cut]
 if past.empty or fut.empty:
 continue
 active = set(past["srcnode"]) | set(past["dstnode"])
 seeds = sorted(mal_a & active)
 if not seeds:
 continue
 fut_max = node_max_loss(fut)
 ice_all = ice_table_corrected(past, set(seeds))
 ci_edge = (ice_all.groupby("edge_idx")["ice"].mean()
 if not ice_all.empty else pd.Series(dtype=float))
 ci_s = ci_sum_from_edges(ci_edge, past)
 sc, nd = ca_apply(fut_max, ci_s, exclude=set(seeds))
 ev = eval_scores(sc, nd, node_to_attacks, all_mal)
 bl_nodes = [n for n in fut_max.index if n not in set(seeds)]
 ev_bl = eval_scores(fut_max.reindex(bl_nodes).values, bl_nodes, node_to_attacks, all_mal)
 xres[a] = {"n_seeds": len(seeds), "baseline_adp": ev_bl["adp"], "ca_adp": ev["adp"],
 "p@10": ev["p@10"], "fut_mal_in_eval": ev["n_mal_eval"]}
 print(f"cross-attack {a}: seeds={len(seeds)} bl={ev_bl['adp']} ca={ev['adp']}", flush=True)
 out["cross_attack"] = xres
 out["grid_rows"] = rows

 with open(os.path.join(OUT, f"e12_{tag}.json"), "w") as f:
 json.dump(out, f, indent=1)
 print(f"saved {OUT}/grid_{tag}.csv + e12_{tag}.json", flush=True)

if __name__ == "__main__":
 main()
