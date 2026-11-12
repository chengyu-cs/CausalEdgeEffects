#!/usr/bin/env python3
"""E1b: four-arm deletion control + killer permutations on the loss table.

Arms (per q in {0.1%, 0.5%, 1%, 5%} of edges removed):
 - top-q% : remove highest-loss edges (the R0 reversal arm)
 - random : remove same count of uniformly random edges (20 seeds -> envelope)
 - degmatch : remove same count of edges matched to the top arm on endpoint degree
 (5 seeds, bin by d(src)+d(dst), exclude top-q% edges)
 - bottom-q% : remove lowest-loss edges

Killer experiments (Q12):
 - loss-shuffle : permute the loss column over edge rows (10 seeds), then remove
 top-q% -> does the reversal survive without loss<->edge coupling?
 - epoch-swap : use loss column from a different epoch of the same run, aligned
 by row key, then remove top-q%.

Reports for every arm: absolute ADP, delta vs baseline, ratio vs baseline.
Outputs: results/causal_necessity_removal/<tag>_arms.csv (+ json summary).
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
OUT_DIR = os.path.join(WS, "results/causal_necessity_removal")
QS = (0.001, 0.005, 0.01, 0.05)

DATASET_ATTACKS = {
 "CADETS_E3": {
 "Backdoor_06": "E3-CADETS/node_Nginx_Backdoor_06.csv",
 "Backdoor_12": "E3-CADETS/node_Nginx_Backdoor_12.csv",
 "Backdoor_13": "E3-CADETS/node_Nginx_Backdoor_13.csv",
 },
 "THEIA_E3": {
 "Browser_Extension_Drakon": "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
 "Firefox_Backdoor_Drakon": "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
 },
}

def compute_adp(scores, nodes, node2attacks, labels):
 scores = np.asarray(scores, float)
 labels = np.asarray(labels, int)
 order = np.argsort(scores)[::-1]
 total_attacks = len({a for atks in node2attacks.values() for a in atks})
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

def node_max_loss(df):
 g = pd.concat([df.groupby("srcnode")["loss"].max(), df.groupby("dstnode")["loss"].max()])
 return g.groupby(level=0).max()

def adp_after_removal(df, mask_remove):
 kept = df[~mask_remove]
 nm = node_max_loss(kept)
 nodes = list(nm.index)
 labels = [1 if n in ALL_MAL else 0 for n in nodes]
 return compute_adp(nm.values, nodes, NODE2ATTACKS, labels)

def load_epoch_df(epoch_dir):
 files = sorted(glob.glob(os.path.join(epoch_dir, "*.csv")))
 with ThreadPoolExecutor(max_workers=min(48, len(files))) as ex:
 dfs = list(ex.map(pd.read_csv, files))
 return pd.concat(dfs, ignore_index=True)

def degree_series(df):
 deg = pd.concat([df["srcnode"], df["dstnode"]]).value_counts()
 return df["srcnode"].map(deg).fillna(0) + df["dstnode"].map(deg).fillna(0)

def run_tag(h, dataset, ep):
 global ALL_MAL, NODE2ATTACKS
 NODE2ATTACKS, ALL_MAL = {}, set()
 for a, f in DATASET_ATTACKS[dataset].items():
 gtdf = pd.read_csv(os.path.join(GT_BASE, f), header=None, names=["uuid", "label", "index_id"])
 ids = set(gtdf["index_id"].astype(int))
 ALL_MAL |= ids
 for n in ids:
 NODE2ATTACKS.setdefault(n, set()).add(a)

 tag = f"{h[:8]}_{dataset}_ep{ep}"
 p = os.path.join(TRAIN, h, dataset, "edge_losses", "test", f"model_epoch_{ep}")
 df = load_epoch_df(p)
 deg = degree_series(df)
 base = adp_after_removal(df, np.zeros(len(df), dtype=bool))
 print(f"[{tag}] n_edges={len(df)} baseline ADP={base:.4f}", flush=True)

 rows = [{"arm": "baseline", "q": None, "seed": None, "adp": base,
 "delta": 0.0, "ratio": 1.0, "n_rm": 0, "rm_mal_frac": 0.0}]

 top_idx = {}
 for q in QS:
 thr = df["loss"].quantile(1 - q)
 m = (df["loss"] > thr).to_numpy()
 top_idx[q] = np.where(m)[0]
 n_rm = int(m.sum())
 # malicious fraction among removed (edges touching a malicious node)
 touch_mal = df["srcnode"].isin(ALL_MAL) | df["dstnode"].isin(ALL_MAL)
 rm_mal = float(touch_mal.to_numpy()[m].mean()) if n_rm else 0.0

 adp = adp_after_removal(df, m)
 rows.append(dict(arm="top", q=q, seed=None, adp=adp, delta=adp - base,
 ratio=adp / base if base > 0 else np.nan, n_rm=n_rm, rm_mal_frac=rm_mal))

 mb = (df["loss"] <= df["loss"].quantile(q)).to_numpy()
 adp = adp_after_removal(df, mb)
 rows.append(dict(arm="bottom", q=q, seed=None, adp=adp, delta=adp - base,
 ratio=adp / base if base > 0 else np.nan, n_rm=int(mb.sum()),
 rm_mal_frac=float(touch_mal.to_numpy()[mb].mean()) if mb.sum() else 0.0))

 rng = np.random.default_rng(1000 + int(q * 10000))
 for s in range(20):
 mr = np.zeros(len(df), dtype=bool)
 mr[rng.choice(len(df), size=n_rm, replace=False)] = True
 adp = adp_after_removal(df, mr)
 rows.append(dict(arm="random", q=q, seed=s, adp=adp, delta=adp - base,
 ratio=adp / base if base > 0 else np.nan, n_rm=n_rm,
 rm_mal_frac=float(touch_mal.to_numpy()[mr].mean())))

 # degree-matched: bin by degree log-quantiles, sample same-bin edges excluding top arm
 topd = pd.Series(deg.to_numpy()[top_idx[q]])
 bins = pd.qcut(np.log1p(deg.to_numpy()), 20, labels=False, duplicates="drop")
 top_bins = pd.Series(bins[top_idx[q]]).value_counts()
 rng2 = np.random.default_rng(2000 + int(q * 10000))
 not_top = np.setdiff1d(np.arange(len(df)), top_idx[q])
 nb = bins[not_top]
 for s in range(5):
 chosen = []
 for b, cnt in top_bins.items():
 cand = not_top[nb == b]
 take = min(len(cand), int(cnt))
 chosen.append(rng2.choice(cand, size=take, replace=False))
 md = np.zeros(len(df), dtype=bool)
 md[np.concatenate(chosen)] = True
 adp = adp_after_removal(df, md)
 rows.append(dict(arm="degmatch", q=q, seed=s, adp=adp, delta=adp - base,
 ratio=adp / base if base > 0 else np.nan, n_rm=int(md.sum()),
 rm_mal_frac=float(touch_mal.to_numpy()[md].mean())))
 print(f"[{tag}] q={q}: arms done (top n_rm={n_rm})", flush=True)

 # ---- killer 1: loss-shuffle ----
 rng = np.random.default_rng(42)
 base_loss = df["loss"].to_numpy()
 for s in range(10):
 df_sh = df.copy()
 df_sh["loss"] = rng.permutation(base_loss)
 nm = node_max_loss(df_sh)
 nodes = list(nm.index)
 labels = [1 if n in ALL_MAL else 0 for n in nodes]
 b_sh = compute_adp(nm.values, nodes, NODE2ATTACKS, labels)
 for q in (0.001, 0.01):
 thr = df_sh["loss"].quantile(1 - q)
 m = (df_sh["loss"] > thr).to_numpy()
 adp = adp_after_removal(df_sh, m)
 rows.append(dict(arm="shuffle_top", q=q, seed=s, adp=adp, delta=adp - b_sh,
 ratio=adp / b_sh if b_sh > 0 else np.nan,
 n_rm=int(m.sum()), rm_mal_frac=np.nan))
 rows.append(dict(arm="shuffle_baseline", q=q, seed=s, adp=b_sh, delta=0.0,
 ratio=1.0, n_rm=0, rm_mal_frac=np.nan))
 print(f"[{tag}] killer shuffle seed {s} done", flush=True)

 out = pd.DataFrame(rows)
 os.makedirs(OUT_DIR, exist_ok=True)
 out.to_csv(os.path.join(OUT_DIR, f"{tag}_arms.csv"), index=False)
 print(f"[{tag}] wrote {len(out)} rows", flush=True)
 return out

def run_epoch_swap(h, dataset, ep_main, ep_other):
 """Killer 2: swap the loss column from another epoch (same run), aligned by row key.

 2026-10-09 fix (/): the original 3-key merge (src,dst,time)
 inflated the two CADETS runs 2.00x (675,069 duplicated 3-keys -> 3,985,667
 rows from 1,987,981). Alignment now uses 4 keys + occurrence rank within key
 (strict one-to-one). The archived *_swaploss_*_arms.csv for 266a87dd and
 ea29a690 were regenerated as *_swap4k_*_arms.csv by e2c_swap4k_envelope.py;
 the kairos swap numbers were unchanged (3 keys were already unique there).
 """
 global ALL_MAL, NODE2ATTACKS
 NODE2ATTACKS, ALL_MAL = {}, set()
 for a, f in DATASET_ATTACKS[dataset].items():
 gtdf = pd.read_csv(os.path.join(GT_BASE, f), header=None, names=["uuid", "label", "index_id"])
 ids = set(gtdf["index_id"].astype(int))
 ALL_MAL |= ids
 for n in ids:
 NODE2ATTACKS.setdefault(n, set()).add(a)
 tag = f"{h[:8]}_{dataset}_ep{ep_main}_swap4k_ep{ep_other}"
 p1 = os.path.join(TRAIN, h, dataset, "edge_losses", "test", f"model_epoch_{ep_main}")
 p2 = os.path.join(TRAIN, h, dataset, "edge_losses", "test", f"model_epoch_{ep_other}")
 df1, df2 = load_epoch_df(p1), load_epoch_df(p2)
 key = ["srcnode", "dstnode", "time", "edge_type"]
 df1["occ"] = df1.groupby(key).cumcount()
 df2["occ"] = df2.groupby(key).cumcount()
 m = df1[key + ["occ", "loss"]].merge(df2[key + ["occ", "loss"]], on=key + ["occ"],
 how="inner", suffixes=("", "_other"))
 print(f"[{tag}] aligned rows: {len(m)} / {len(df1)} (inflate {len(m)/len(df1):.3f}x)",
 flush=True)
 df_sw = m.copy()
 df_sw["loss"] = m["loss_other"]
 base = adp_after_removal(df_sw, np.zeros(len(df_sw), dtype=bool))
 rows = [{"arm": "swap_baseline", "q": None, "seed": None, "adp": base, "delta": 0.0,
 "ratio": 1.0, "n_rm": 0, "rm_mal_frac": 0.0}]
 for q in QS:
 thr = df_sw["loss"].quantile(1 - q)
 mk = (df_sw["loss"] > thr).to_numpy()
 adp = adp_after_removal(df_sw, mk)
 rows.append(dict(arm="swap_top", q=q, seed=None, adp=adp, delta=adp - base,
 ratio=adp / base if base > 0 else np.nan, n_rm=int(mk.sum()),
 rm_mal_frac=np.nan))
 print(f"[{tag}] swap_top q={q}: {adp:.4f} (base {base:.4f})", flush=True)
 out = pd.DataFrame(rows)
 out.to_csv(os.path.join(OUT_DIR, f"{tag}_arms.csv"), index=False)
 return out

if __name__ == "__main__":
 # run-hash sync: 266a87dd is ORTHRUS CADETS (not
 # velox; Stage4/Server_Info table had the two CADETS cells swapped, see
 # run-hash mapping). ea29a690 is VELUX CADETS. Its arms
 # were generated by: python removal_control_arms.py ea29a690 (extra branch below).
 HASHES = {
 "kairos_THEIA": ("1a43f8e8253beb62033f23193471422a0a96647e4113d1b6e1faae4f91697505", "THEIA_E3"),
 "orthrus_CADETS": ("266a87dde3fae4c72f17282bc10c4ad1cecef69c55f8e47796d65fedc1601de9", "CADETS_E3"),
 "velox_CADETS": ("ea29a6902c6651cf2308eaa76e0523772454ce42f4c50e77cd99c932e211ce4a", "CADETS_E3"),
 }
 which = sys.argv[1] if len(sys.argv) > 1 else "all"
 if which in ("all", "kairos"):
 h, ds = HASHES["kairos_THEIA"]
 run_tag(h, ds, 19)
 run_epoch_swap(h, ds, 19, 11)
 if which in ("all", "orthrus"):
 h, ds = HASHES["orthrus_CADETS"]
 run_tag(h, ds, 13)
 run_epoch_swap(h, ds, 13, 19)
 if which in ("all", "velox"):
 h, ds = HASHES["velox_CADETS"]
 run_tag(h, ds, 13)
 run_epoch_swap(h, ds, 13, 19)
 print("E1B ALL DONE")
