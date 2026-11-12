"""Stage 3: Causal-Aware Detection + Attack Critical Path.

Extends the Stage 2 Causal Screening prototype with two full research methods:

 Task A - Causal-Aware Detection:
 Amplify anomaly signals using Causal Importance (CI). For each node v,
 the baseline Orthrus score (max edge loss) is multiplied by a sigmoid
 gating term driven by the sum of CI over edges touching v. Malicious
 nodes retain ~full score while benign nodes are suppressed, improving
 detection precision.

 CA(v) = baseline(v) * sigmoid(alpha * ci_sum(v))

 Task B - Attack Critical Path:
 For each attack scenario, build a directed subgraph of the malicious
 neighborhood, weight edges by CI, and find the maximum-weight path
 from the attack entry node to the top-scoring anomaly node. This
 path represents the causally most important attack progression route.

 ACP-IoU:
 New evaluation metric comparing the identified critical path against
 ground-truth edges (edges with >=1 malicious endpoint). IoU of the
 edge sets measures path quality.

Usage:
 python causal_edge_effect_pipeline.py orthrus CADETS_E3
 python causal_edge_effect_pipeline.py orthrus CADETS_E3 --epoch 15
 python causal_edge_effect_pipeline.py orthrus CADETS_E3 --alpha 2.0

Requirements:
 - Training must have been run (edge_losses CSVs exist in artifacts)
 - networkx for path finding (pip install networkx)
 - No model loading or CUDA needed - runs on CPU
"""

import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# Force line-buffered stdout so log output appears immediately
if hasattr(sys.stdout, "reconfigure"):
 sys.stdout.reconfigure(line_buffering=True)

# Prevent nltk from trying to download data at import time (server may lack internet)
import nltk
nltk.download = lambda *a, **k: True

import numpy as np
import pandas as pd

from pidsmaker.config import get_runtime_required_args, get_yml_cfg

def log(msg: str, *args, **kwargs):
 """Local log function with flush (avoids pidsmaker.utils.utils which triggers nltk)."""
 from datetime import datetime as _dt
 timestamp = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
 print(f"{timestamp} - {msg}", *args, **kwargs, flush=True)

# =============================================================================
# Attack ground-truth definitions (from config.py + Ground_Truth/orthrus/)
# =============================================================================
DATASET_ATTACKS = {
 "CADETS_E3": {
 "Backdoor_06": {"gt_file": "E3-CADETS/node_Nginx_Backdoor_06.csv", "tw_start": "2018-04-06 11:20:00", "tw_end": "2018-04-06 12:09:00"},
 "Backdoor_12": {"gt_file": "E3-CADETS/node_Nginx_Backdoor_12.csv", "tw_start": "2018-04-12 13:59:00", "tw_end": "2018-04-12 14:39:00"},
 "Backdoor_13": {"gt_file": "E3-CADETS/node_Nginx_Backdoor_13.csv", "tw_start": "2018-04-13 09:03:00", "tw_end": "2018-04-13 09:16:00"},
 },
 "CADETS_E5": {
 "Nginx_Drakon_APT": {"gt_file": "E5-CADETS/node_Nginx_Drakon_APT.csv", "tw_start": "2019-05-16 09:31:00", "tw_end": "2019-05-16 10:12:00"},
 "Nginx_Drakon_APT_17": {"gt_file": "E5-CADETS/node_Nginx_Drakon_APT_17.csv", "tw_start": "2019-05-17 10:15:00", "tw_end": "2019-05-17 15:33:00"},
 },
 "THEIA_E3": {
 "Browser_Extension_Drakon": {"gt_file": "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv", "tw_start": "2018-04-12 12:40:00", "tw_end": "2018-04-12 13:30:00"},
 "Firefox_Backdoor_Drakon": {"gt_file": "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv", "tw_start": "2018-04-10 14:30:00", "tw_end": "2018-04-10 15:00:00"},
 },
 "THEIA_E5": {
 "Firefox_Drakon_APT": {"gt_file": "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv", "tw_start": "2019-05-15 14:47:00", "tw_end": "2019-05-15 15:08:00"},
 },
 "CLEARSCOPE_E5": {
 "AppStarter_0515": {"gt_file": "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv", "tw_start": "2019-05-15 15:38:00", "tw_end": "2019-05-15 16:19:00"},
 "Tester_0517": {"gt_file": "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv", "tw_start": "2019-05-17 16:20:00", "tw_end": "2019-05-17 16:28:00"},
 },
 "CLEARSCOPE_E3": {
 "Firefox_0411": {"gt_file": "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv", "tw_start": "2018-04-11 13:54:00", "tw_end": "2018-04-11 14:48:00"},
 },
 "optc_h051": {
 "H051_0925": {"gt_file": "h051/node_h051_0925.csv", "tw_start": "2019-09-25 10:29:00", "tw_end": "2019-09-25 14:25:00"},
 },
 "optc_h201": {
 "H201_0923": {"gt_file": "h201/node_h201_0923.csv", "tw_start": "2019-09-23 11:23:00", "tw_end": "2019-09-23 13:25:00"},
 },
 "optc_h501": {
 "H501_0924": {"gt_file": "h501/node_h501_0924.csv", "tw_start": "2019-09-24 10:28:00", "tw_end": "2019-09-24 15:29:00"},
 },
}

GROUND_TRUTH_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Ground_Truth", "orthrus")

# =============================================================================
# Ground-truth loading
# =============================================================================
def load_ground_truth(dataset_name):
 """Load malicious node IDs and labels for all attack scenarios in a dataset.

 Returns:
 attack_to_malicious: {attack_name -> set(node_ids)}
 node_labels: {node_id -> label_string} for readable output
 node_to_attacks: {node_id -> set(attack_names)} (a node may belong to multiple attacks)
 """
 ATTACKS = DATASET_ATTACKS.get(dataset_name)
 if ATTACKS is None:
 raise ValueError(
 f"Dataset '{dataset_name}' not supported. "
 f"Available: {list(DATASET_ATTACKS.keys())}"
 )

 attack_to_malicious = {}
 node_labels = {}
 node_to_attacks = defaultdict(set)

 for attack_name, info in ATTACKS.items():
 gt_path = os.path.join(GROUND_TRUTH_BASE, info["gt_file"])
 df = pd.read_csv(gt_path, header=None, names=["uuid", "label", "index_id"])
 ids = set(df["index_id"].astype(int).tolist())
 attack_to_malicious[attack_name] = ids
 for _, row in df.iterrows():
 nid = int(row["index_id"])
 node_labels[nid] = row["label"]
 node_to_attacks[nid].add(attack_name)
 log(f" {attack_name}: {len(ids)} malicious nodes ({info['tw_start']} ~ {info['tw_end']})")

 node_to_attacks = dict(node_to_attacks)
 return attack_to_malicious, node_labels, node_to_attacks

# =============================================================================
# Edge-loss loading (reused from Stage 2 prototype)
# =============================================================================
def find_edge_losses_dir(cfg):
 """Find the edge_losses/test directory across all config-hash directories."""
 artifact_dir = cfg._artifact_dir
 dataset_name = cfg.dataset.name
 pattern = os.path.join(
 artifact_dir, "training", "training", "*", dataset_name, "edge_losses", "test"
 )
 matches = glob.glob(pattern)
 if not matches:
 raise FileNotFoundError(
 f"No edge_losses/test directory found.\n"
 f"Searched: {pattern}\n"
 f"Run training first: python pidsmaker/main.py orthrus {dataset_name} "
 f"--training.lr=0.001 --training.num_epochs=20 --force_restart training"
 )
 matches.sort(key=os.path.getmtime, reverse=True)
 return matches[0]

def list_available_epochs(edge_losses_dir):
 epochs = []
 for entry in os.listdir(edge_losses_dir):
 if entry.startswith("model_epoch_"):
 try:
 epochs.append(int(entry.replace("model_epoch_", "")))
 except ValueError:
 pass
 epochs.sort()
 return epochs

def load_edge_losses(edge_losses_dir, epoch=None):
 """Load all edge losses from CSV files for a given epoch (parallel I/O).

 Returns:
 list of dicts with keys: src, dst, t, loss, edge_type
 """
 if epoch is None:
 epochs = list_available_epochs(edge_losses_dir)
 if not epochs:
 raise FileNotFoundError(f"No model_epoch_* directories in {edge_losses_dir}")
 epoch = epochs[-1]
 log(f"Using latest epoch: {epoch} (available: {epochs})")
 else:
 log(f"Using specified epoch: {epoch}")

 epoch_dir = os.path.join(edge_losses_dir, f"model_epoch_{epoch}")
 if not os.path.isdir(epoch_dir):
 raise FileNotFoundError(f"Epoch directory not found: {epoch_dir}")

 csv_files = sorted(glob.glob(os.path.join(epoch_dir, "*.csv")))
 n_workers = min(os.cpu_count() or 4, len(csv_files))
 log(f"Loading {len(csv_files)} time-window CSV files from {epoch_dir} ({n_workers} threads)")

 with ThreadPoolExecutor(max_workers=n_workers) as executor:
 dfs = list(executor.map(pd.read_csv, csv_files))

 df = pd.concat(dfs, ignore_index=True)
 df = df.rename(columns={"srcnode": "src", "dstnode": "dst", "time": "t"})
 all_edges = df.to_dict("records")

 log(f"Loaded {len(all_edges)} test edges")
 return all_edges, epoch

# =============================================================================
# Node scoring and CI computation (vectorized)
# =============================================================================
def compute_ci_scores(all_edges, all_malicious_ids, use_dst_loss=True):
 """Compute Causal Importance (CI) for every edge via aggregation-level ICE.

 Vectorized implementation using pandas/numpy. For the ``max_val_loss``
 aggregation, ICE has a closed form:

 If edge e's loss is the unique maximum for node v:
 ICE(e, v) = max_loss(v) - second_max_loss(v) (or max_loss if v has only 1 edge)
 Otherwise:
 ICE(e, v) = 0 (removing e doesn't change the max)

 CI(e) = mean(|ICE(e, v)|) over malicious endpoints of e.
 Edges touching no malicious node have CI = 0.

 Returns:
 ci_scores: {(src, dst, t) -> float}
 node_scores: {node -> float}
 """
 malicious_set = set(all_malicious_ids)
 n_edges = len(all_edges)

 df = pd.DataFrame(all_edges)
 # Ensure integer types for node IDs
 df["src"] = df["src"].astype(np.int64)
 df["dst"] = df["dst"].astype(np.int64)
 df["t"] = df["t"].astype(np.int64)

 # --- Build long-format: one row per (node, edge) pair ---
 src_part = pd.DataFrame({
 "node": df["src"], "loss": df["loss"], "edge_idx": np.arange(n_edges)
 })
 if use_dst_loss:
 dst_part = pd.DataFrame({
 "node": df["dst"], "loss": df["loss"], "edge_idx": np.arange(n_edges)
 })
 long_df = pd.concat([src_part, dst_part], ignore_index=True)
 else:
 long_df = src_part

 # --- Per-node stats via groupby (vectorized) ---
 # max_loss per node
 node_max = long_df.groupby("node")["loss"].max()

 # second_max per node
 # Corrected 2026-10-11: the previous implementation used
 # `sort_values(...).groupby("node").head(2).groupby("node")["loss"].nth(1)`,
 # whose nth(1) returns a Series indexed by ORIGINAL ROW LABELS (pandas
 # semantics), not by node id — after reindex(node_max.index) almost every
 # entry became NaN and fillna(0.0) made second_max identically 0, so the
 # closed-form ICE degenerated to max-0 (see results/closed_form_validation/e2a_toy_bug_verification.json
 # and R1 E2a records). The fix follows the independently verified
 # e2a_corrected_closed.py: per-group nlargest(2). Nodes with a single edge
 # keep second=0, matching the pre-correction fillna(0.0) convention.
 node_second = long_df.groupby("node")["loss"].apply(
 lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else 0.0
 )

 # count of edges with max loss per node
 max_per_node = long_df.groupby("node")["loss"].transform("max")
 long_df["is_max"] = long_df["loss"] == max_per_node
 node_count_max = long_df[long_df["is_max"]].groupby("node").size()
 node_count_max = node_count_max.reindex(node_max.index).fillna(0).astype(int)

 # Node scores (baseline)
 node_scores = node_max.to_dict()

 # --- Compute ICE per (node, edge) pair ---
 long_df = long_df.merge(node_max.rename("n_max"), left_on="node", right_index=True)
 long_df = long_df.merge(node_second.rename("n_second"), left_on="node", right_index=True)
 long_df = long_df.merge(node_count_max.rename("n_count_max"), left_on="node", right_index=True)

 # leave_one_out_max = second_max if (edge is unique max) else max
 long_df["is_unique_max"] = (long_df["loss"] == long_df["n_max"]) & (long_df["n_count_max"] == 1)
 long_df["loo_max"] = np.where(long_df["is_unique_max"], long_df["n_second"], long_df["n_max"])
 long_df["ice"] = (long_df["n_max"] - long_df["loo_max"]).abs()

 # Zero out ICE for non-malicious nodes
 long_df["is_malicious"] = long_df["node"].isin(malicious_set)
 long_df.loc[~long_df["is_malicious"], "ice"] = 0.0

 # --- CI per edge = mean ICE over malicious endpoints ---
 mal_ice = long_df[long_df["is_malicious"]].groupby("edge_idx")["ice"].mean()
 ci_per_edge = mal_ice.reindex(np.arange(n_edges)).fillna(0.0).values

 # Build ci_scores dict
 src_arr = df["src"].values
 dst_arr = df["dst"].values
 t_arr = df["t"].values
 ci_scores = {
 (int(s), int(d), int(t)): float(c)
 for s, d, t, c in zip(src_arr, dst_arr, t_arr, ci_per_edge)
 }

 ice_count = int((ci_per_edge > 0).sum())
 log(f"Computed CI for {n_edges} edges ({ice_count} with non-zero ICE, "
 f"{n_edges - ice_count} zero) [vectorized]")
 return ci_scores, node_scores

def compute_ci_sum_per_node(ci_scores):
 """Sum CI over all edges touching each node (vectorized).

 Only edges with non-zero CI contribute.
 """
 if not ci_scores:
 return {}

 keys = list(ci_scores.keys())
 ci_arr = np.array(list(ci_scores.values()), dtype=float)
 mask = ci_arr > 0
 if not mask.any():
 return {}

 src_arr = np.array([k[0] for k in keys])[mask]
 dst_arr = np.array([k[1] for k in keys])[mask]
 ci_vals = ci_arr[mask]

 all_nodes = np.concatenate([src_arr, dst_arr])
 all_ci = np.concatenate([ci_vals, ci_vals])

 ci_sum = defaultdict(float)
 for node, ci in zip(all_nodes, all_ci):
 ci_sum[int(node)] += ci
 return dict(ci_sum)

def sigmoid_array(x):
 """Numerically stable sigmoid for numpy arrays (also works for scalars)."""
 x = np.asarray(x, dtype=float)
 out = np.empty_like(x)
 pos = x >= 0
 neg = ~pos
 out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
 ex = np.exp(x[neg])
 out[neg] = ex / (1.0 + ex)
 return out

def compute_ca_scores(node_scores, ci_sum, alpha, bias=0.0):
 """Causal-Aware score: baseline * sigmoid(alpha * ci_sum - bias).

 Vectorized over all nodes.
 - Malicious nodes (high ci_sum): sigmoid -> 1, CA ~= baseline (no suppression)
 - Benign nodes (ci_sum = 0): sigmoid(-bias), suppressed more with higher bias

 bias=0: sigmoid(0)=0.5, 50% suppression (default, used in Task A)
 bias=2: sigmoid(-2)=0.12, 88% suppression (aggressive, for no-GT deployment)
 """
 nodes = list(node_scores.keys())
 baselines = np.array([node_scores[n] for n in nodes], dtype=float)
 ci_sums = np.array([ci_sum.get(n, 0.0) for n in nodes], dtype=float)
 ca_values = baselines * sigmoid_array(alpha * ci_sums - bias)
 return {node: float(ca) for node, ca in zip(nodes, ca_values)}

# =============================================================================
# Task A: Causal-Aware Detection (ADP evaluation)
# =============================================================================
def compute_adp(scores, nodes, node2attacks, labels, return_curve=False):
 """Attack Detection vs Precision (ADP).

 Replicates ``plot_detected_attacks_vs_precision`` from evaluation_utils.py:633
 without the matplotlib dependency. Returns the normalised area under the
 detected-attacks-vs-precision curve (0..1).

 If return_curve=True, also returns the curve data (precision, detected_pct)
 for plotting.
 """
 scores = np.asarray(scores, dtype=float)
 labels = np.asarray(labels, dtype=int)
 sorted_indices = np.argsort(scores)[::-1]
 sorted_nodes = [nodes[i] for i in sorted_indices]
 sorted_labels = labels[sorted_indices]

 total_attacks = len({a for atks in node2attacks.values() for a in atks})
 if total_attacks == 0:
 if return_curve:
 return 0.0, {"precisions": [0.0], "detected_pcts": [0.0]}
 return 0.0

 detected = set()
 tp = 0
 fp = 0
 precisions = [0.0]
 detected_pcts = [0.0]

 for i, node in enumerate(sorted_nodes):
 if sorted_labels[i] == 1:
 tp += 1
 if node in node2attacks:
 detected.update(node2attacks[node])
 else:
 fp += 1
 precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
 pct = (len(detected) / total_attacks) * 100
 precisions.append(precision)
 detected_pcts.append(pct)

 # Collapse duplicates: keep max detected% per precision value
 prec_to_max = {}
 for p, d in zip(precisions, detected_pcts):
 prec_to_max[p] = max(prec_to_max.get(p, 0.0), d)

 sorted_precs = sorted(prec_to_max.keys())
 sorted_dets = [prec_to_max[p] for p in sorted_precs]

 area = float(np.trapz(sorted_dets, sorted_precs) / 100.0)

 if return_curve:
 curve_data = {
 "precisions": [round(p, 6) for p in sorted_precs],
 "detected_pcts": [round(d, 2) for d in sorted_dets],
 "total_attacks": total_attacks,
 "total_nodes": len(nodes),
 "malicious_nodes": int(sum(labels)),
 }
 return area, curve_data
 return area

def compute_precision_at_k(scores, nodes, labels, k=10):
 """Precision when taking the top-k nodes by score."""
 scores = np.asarray(scores, dtype=float)
 sorted_indices = np.argsort(scores)[::-1]
 top_k = min(k, len(sorted_indices))
 tp = int(sum(labels[sorted_indices[:top_k]]))
 return tp / top_k if top_k > 0 else 0.0

def count_detected_attacks(scores, nodes, labels, node2attacks, k=10):
 """How many unique attacks are covered by the top-k nodes."""
 scores = np.asarray(scores, dtype=float)
 sorted_indices = np.argsort(scores)[::-1]
 top_k = min(k, len(sorted_indices))
 detected = set()
 for i in range(top_k):
 node = nodes[sorted_indices[i]]
 if labels[sorted_indices[i]] == 1 and node in node2attacks:
 detected.update(node2attacks[node])
 return len(detected)

def run_task_a(node_scores, ca_scores, node_to_attacks, attack_to_malicious):
 """Compare Baseline vs Causal-Aware detection via ADP / Precision@10 / detected attacks."""
 all_malicious = set()
 for ids in attack_to_malicious.values():
 all_malicious.update(ids)

 nodes = list(node_scores.keys())
 baseline_list = [node_scores[n] for n in nodes]
 ca_list = [ca_scores.get(n, 0.0) for n in nodes]
 labels = [1 if n in all_malicious else 0 for n in nodes]

 total_attacks = len(attack_to_malicious)

 # Baseline metrics (with curve data for plotting)
 base_adp, base_curve = compute_adp(baseline_list, nodes, node_to_attacks, labels, return_curve=True)
 base_p10 = compute_precision_at_k(baseline_list, nodes, np.array(labels), k=10)
 base_detected = count_detected_attacks(baseline_list, nodes, np.array(labels), node_to_attacks, k=10)

 # CA metrics (with curve data for plotting)
 ca_adp, ca_curve = compute_adp(ca_list, nodes, node_to_attacks, labels, return_curve=True)
 ca_p10 = compute_precision_at_k(ca_list, nodes, np.array(labels), k=10)
 ca_detected = count_detected_attacks(ca_list, nodes, np.array(labels), node_to_attacks, k=10)

 results = {
 "baseline": {
 "adp": round(base_adp, 4),
 "precision_at_10": round(base_p10, 4),
 "detected_attacks": f"{base_detected}/{total_attacks}",
 },
 "causal_aware": {
 "adp": round(ca_adp, 4),
 "precision_at_10": round(ca_p10, 4),
 "detected_attacks": f"{ca_detected}/{total_attacks}",
 },
 "improvement": {
 "adp_delta": round(ca_adp - base_adp, 4),
 "precision_delta": round(ca_p10 - base_p10, 4),
 },
 "adp_curves": {
 "baseline": base_curve,
 "causal_aware": ca_curve,
 },
 }

 log("")
 log("--- Task A: Causal-Aware Detection ---")
 log(f"{'':24s} {'Baseline (Orthrus)':>22s} {'Causal-Aware (CA)':>22s}")
 log(f"{'ADP:':24s} {base_adp:>22.4f} {ca_adp:>22.4f}")
 log(f"{'Precision@10:':24s} {base_p10:>22.4f} {ca_p10:>22.4f}")
 log(f"{'Detected attacks:':24s} {f'{base_detected}/{total_attacks}':>22s} {f'{ca_detected}/{total_attacks}':>22s}")
 delta = ca_adp - base_adp
 tag = "IMPROVED" if delta > 0 else ("NO CHANGE" if delta == 0 else "DEGRADED")
 log(f"\n ADP delta = {delta:+.4f} [{tag}]")

 return results

# =============================================================================
# Experiment 1: Causal Necessity Validation
# =============================================================================
def causal_necessity_experiment(all_edges, ci_scores, node_scores, node_to_attacks, all_malicious, use_dst_loss=True):
 """Experiment 1: Remove top-K edges by different strategies, measure ADP change.

 If CI identifies causally necessary edges, removing them should crash ADP,
 while removing random or high-loss edges should have minimal effect.
 """
 import random

 nodes = list(node_scores.keys())
 labels_list = [1 if n in all_malicious else 0 for n in nodes]
 baseline_adp = compute_adp([node_scores[n] for n in nodes], nodes, node_to_attacks, labels_list)

 log("")
 log("=" * 72)
 log("--- Experiment 1: Causal Necessity ---")
 log(f"Baseline ADP (no removal): {baseline_adp:.4f}")

 # Prepare sorted edge lists
 ci_sorted = sorted(ci_scores.items(), key=lambda x: x[1], reverse=True)
 loss_sorted = sorted(all_edges, key=lambda e: e["loss"], reverse=True)
 random.seed(42)
 random_edges = list(random.sample(range(len(all_edges)), min(50, len(all_edges))))

 ks = [5, 10, 20, 50]
 results = {"baseline_adp": round(baseline_adp, 4), "removals": []}

 # Header
 log(f"\n{'K':>5s} {'CI (causal)':>15s} {'Loss (correl.)':>15s} {'Random':>15s}")
 log("-" * 55)

 for K in ks:
 row = {"K": K, "strategies": []}
 adp_values = []

 for strategy_name, removed_keys in [
 ("CI", set((k[0], k[1], k[2]) for k, _ in ci_sorted[:K])),
 ("Loss", set((e["src"], e["dst"], e["t"]) for e in loss_sorted[:K])),
 ("Random", set((all_edges[idx]["src"], all_edges[idx]["dst"], all_edges[idx]["t"]) for idx in random_edges[:K])),
 ]:
 # Recompute node scores without removed edges
 node_to_losses_new = defaultdict(list)
 for e in all_edges:
 key = (e["src"], e["dst"], e["t"])
 if key in removed_keys:
 continue
 node_to_losses_new[e["src"]].append(e["loss"])
 if use_dst_loss:
 node_to_losses_new[e["dst"]].append(e["loss"])

 new_scores = [max(losses) if losses else 0.0 for n in nodes for losses in [node_to_losses_new.get(n, [])]]
 adp = compute_adp(new_scores, nodes, node_to_attacks, labels_list)
 delta = adp - baseline_adp
 adp_values.append(adp)
 row["strategies"].append({"strategy": strategy_name, "adp": round(adp, 4), "delta": round(delta, 4)})

 results["removals"].append(row)
 log(f"{K:>5d} {adp_values[0]:>10.4f} (Δ={adp_values[0]-baseline_adp:+.4f}) "
 f"{adp_values[1]:>10.4f} (Δ={adp_values[1]-baseline_adp:+.4f}) "
 f"{adp_values[2]:>10.4f} (Δ={adp_values[2]-baseline_adp:+.4f})")

 # Summary
 ci_drops = [r["strategies"][0]["delta"] for r in results["removals"]]
 loss_drops = [r["strategies"][1]["delta"] for r in results["removals"]]
 random_drops = [r["strategies"][2]["delta"] for r in results["removals"]]
 log(f"\n CI edges cause {abs(ci_drops[-1]):.4f} ADP drop (K=50)")
 log(f" Loss edges cause {abs(loss_drops[-1]):.4f} ADP drop (K=50)")
 log(f" Random edges cause {abs(random_drops[-1]):.4f} ADP drop (K=50)")

 if abs(ci_drops[-1]) > abs(loss_drops[-1]) * 2:
 log(" >>> CI edges are causally necessary: removing them crashes ADP. <<<")
 else:
 log(" >>> WARNING: CI edges not more impactful than loss edges. <<<")

 return results

# =============================================================================
# Experiment 2: False Positive Analysis
# =============================================================================
def false_positive_analysis(node_scores, ca_scores, ci_sum, all_malicious, top_k=100):
 """Experiment 2: Compare suppression rates for false positives vs true malicious.

 If CA works correctly:
 - False positives (high-score benign) should have ci_sum ≈ 0 → CA ≈ 0.5×baseline → ~50% suppression
 - True malicious should have ci_sum > 0 → CA ≈ baseline → ~0% suppression
 """
 malicious_set = set(all_malicious)

 # Sort all nodes by baseline score descending
 sorted_nodes = sorted(node_scores.keys(), key=lambda n: node_scores[n], reverse=True)

 # Top-K false positives: highest baseline scores among benign nodes
 false_positives = [n for n in sorted_nodes if n not in malicious_set][:top_k]
 # True malicious nodes (that have scores)
 true_malicious = [n for n in sorted_nodes if n in malicious_set]

 def analyze_group(nodes_list, label):
 if not nodes_list:
 return {"label": label, "count": 0, "avg_baseline": 0, "avg_ca": 0, "avg_ci_sum": 0, "avg_suppression": 0}
 baselines = [node_scores[n] for n in nodes_list]
 cas = [ca_scores.get(n, 0.0) for n in nodes_list]
 cis = [ci_sum.get(n, 0.0) for n in nodes_list]
 suppressions = [1.0 - (ca_scores.get(n, 0.0) / max(node_scores[n], 1e-9)) for n in nodes_list]
 return {
 "label": label,
 "count": len(nodes_list),
 "avg_baseline": round(float(np.mean(baselines)), 4),
 "avg_ca": round(float(np.mean(cas)), 4),
 "avg_ci_sum": round(float(np.mean(cis)), 4),
 "avg_suppression": round(float(np.mean(suppressions)), 4),
 }

 fp_stats = analyze_group(false_positives, "False Positives (top-%d benign)" % top_k)
 mal_stats = analyze_group(true_malicious, "True Malicious")

 # Detailed top-10 false positives for inspection
 fp_details = []
 for n in false_positives[:10]:
 fp_details.append({
 "node_id": int(n),
 "baseline": round(node_scores[n], 4),
 "ca_score": round(ca_scores.get(n, 0.0), 4),
 "ci_sum": round(ci_sum.get(n, 0.0), 4),
 "suppression": round(1.0 - ca_scores.get(n, 0.0) / max(node_scores[n], 1e-9), 4),
 })

 # Detailed top-10 malicious for comparison
 mal_details = []
 for n in true_malicious[:10]:
 mal_details.append({
 "node_id": int(n),
 "baseline": round(node_scores[n], 4),
 "ca_score": round(ca_scores.get(n, 0.0), 4),
 "ci_sum": round(ci_sum.get(n, 0.0), 4),
 "suppression": round(1.0 - ca_scores.get(n, 0.0) / max(node_scores[n], 1e-9), 4),
 })

 results = {
 "false_positives": fp_stats,
 "true_malicious": mal_stats,
 "fp_details_top10": fp_details,
 "mal_details_top10": mal_details,
 }

 log("")
 log("=" * 72)
 log("--- Experiment 2: False Positive Analysis ---")
 log(f"\n{'':30s} {'False Positives':>18s} {'True Malicious':>18s}")
 log(f"{'Count':30s} {fp_stats['count']:>18d} {mal_stats['count']:>18d}")
 log(f"{'Avg baseline score':30s} {fp_stats['avg_baseline']:>18.4f} {mal_stats['avg_baseline']:>18.4f}")
 log(f"{'Avg CA score':30s} {fp_stats['avg_ca']:>18.4f} {mal_stats['avg_ca']:>18.4f}")
 log(f"{'Avg ci_sum':30s} {fp_stats['avg_ci_sum']:>18.4f} {mal_stats['avg_ci_sum']:>18.4f}")
 log(f"{'Avg suppression rate':30s} {fp_stats['avg_suppression']:>18.1%} {mal_stats['avg_suppression']:>18.1%}")

 log(f"\n Top-5 False Positives (high-score benign nodes):")
 for d in fp_details[:5]:
 log(f" node {d['node_id']:>8d}: baseline={d['baseline']:.4f}, CA={d['ca_score']:.4f}, "
 f"ci_sum={d['ci_sum']:.4f}, suppression={d['suppression']:.1%}")

 log(f"\n Top-5 True Malicious nodes:")
 for d in mal_details[:5]:
 log(f" node {d['node_id']:>8d}: baseline={d['baseline']:.4f}, CA={d['ca_score']:.4f}, "
 f"ci_sum={d['ci_sum']:.4f}, suppression={d['suppression']:.1%}")

 ratio = fp_stats["avg_suppression"] / max(mal_stats["avg_suppression"], 0.001)
 log(f"\n False positive suppression is {ratio:.1f}× higher than malicious suppression")
 if fp_stats["avg_suppression"] > 0.3 and mal_stats["avg_suppression"] < 0.1:
 log(" >>> CA effectively suppresses false positives while preserving true attacks. <<<")
 else:
 log(" >>> CA suppression effect is weak. <<<")

 return results

# =============================================================================
# Experiment 3: No Ground Truth Deployment
# =============================================================================
def no_gt_deployment_experiment(all_edges, node_scores, node_to_attacks, all_malicious,
 ca_scores_gt, use_dst_loss=True, top_k=100,
 n_iterations=3, bias=2.0):
 """Experiment 3: Compute CI using top-K anomaly nodes (no ground truth).

 Improved with:
 1. Aggressive bias: sigmoid(alpha*ci_sum - bias) suppresses ci_sum=0 nodes to ~12%
 2. Iterative refinement: re-rank by CA score, update reference set, repeat
 """
 nodes = list(node_scores.keys())
 labels_list = [1 if n in all_malicious else 0 for n in nodes]
 labels_arr = np.array(labels_list)

 # Baseline and GT metrics
 baseline_adp = compute_adp([node_scores[n] for n in nodes], nodes, node_to_attacks, labels_list)
 ca_gt_adp = compute_adp([ca_scores_gt.get(n, 0.0) for n in nodes], nodes, node_to_attacks, labels_list)
 base_p10 = compute_precision_at_k([node_scores[n] for n in nodes], nodes, labels_arr, k=10)
 ca_gt_p10 = compute_precision_at_k([ca_scores_gt.get(n, 0.0) for n in nodes], nodes, labels_arr, k=10)

 log("")
 log("=" * 72)
 log("--- Experiment 3: No Ground Truth Deployment (Iterative) ---")
 log(f"Parameters: top_k={top_k}, bias={bias}, iterations={n_iterations}")
 log(f" (bias={bias} suppresses ci_sum=0 nodes to {1/(1+np.exp(bias)):.0%} of baseline)")

 # Iteration 0: reference = top-K by baseline score
 sorted_nodes = sorted(node_scores.keys(), key=lambda n: node_scores[n], reverse=True)
 reference_set = set(sorted_nodes[:top_k])
 mal_in_ref = reference_set & all_malicious

 log(f"\n Iteration 0 (baseline ranking):")
 log(f" top-{top_k} malicious: {len(mal_in_ref)}/{top_k} ({len(mal_in_ref)/top_k:.0%})")

 iteration_results = []

 for iteration in range(n_iterations):
 # Step 1: Compute CI using current reference set
 ci_scores_nogt, _ = compute_ci_scores(all_edges, reference_set, use_dst_loss=use_dst_loss)

 # Step 2: Compute ci_sum and CA scores (with aggressive bias)
 ci_sum_nogt = compute_ci_sum_per_node(ci_scores_nogt)
 ca_scores_nogt = compute_ca_scores(node_scores, ci_sum_nogt, alpha=1.0, bias=bias)

 # Step 3: Evaluate
 ca_nogt_adp = compute_adp([ca_scores_nogt.get(n, 0.0) for n in nodes], nodes, node_to_attacks, labels_list)
 ca_nogt_p10 = compute_precision_at_k([ca_scores_nogt.get(n, 0.0) for n in nodes], nodes, labels_arr, k=10)

 # Step 4: Update reference set (re-rank by CA score)
 sorted_by_ca = sorted(node_scores.keys(), key=lambda n: ca_scores_nogt.get(n, 0.0), reverse=True)
 new_reference_set = set(sorted_by_ca[:top_k])
 new_mal_in_ref = new_reference_set & all_malicious

 log(f"\n Iteration {iteration + 1}:")
 log(f" top-{top_k} malicious: {len(new_mal_in_ref)}/{top_k} ({len(new_mal_in_ref)/top_k:.0%})")
 log(f" CA ADP: {ca_nogt_adp:.4f} Prec@10: {ca_nogt_p10:.4f}")

 iteration_results.append({
 "iteration": iteration + 1,
 "malicious_in_topk": len(new_mal_in_ref),
 "reference_precision": round(len(new_mal_in_ref) / top_k, 4),
 "ca_adp": round(ca_nogt_adp, 4),
 "ca_p10": round(ca_nogt_p10, 4),
 })

 reference_set = new_reference_set

 # Final results
 final = iteration_results[-1]
 ca_nogt_adp = final["ca_adp"]
 ca_nogt_p10 = final["ca_p10"]

 log(f"\n {'':25s} {'ADP':>10s} {'Prec@10':>10s}")
 log(f" {'Baseline':25s} {baseline_adp:>10.4f} {base_p10:>10.4f}")
 log(f" {'CA (with GT)':25s} {ca_gt_adp:>10.4f} {ca_gt_p10:>10.4f}")
 log(f" {'CA (no GT, iter=%d)' % n_iterations:25s} {ca_nogt_adp:>10.4f} {ca_nogt_p10:>10.4f}")

 nogt_delta = ca_nogt_adp - baseline_adp
 log(f"\n No-GT CA improvement over baseline: {nogt_delta:+.4f}")
 log(f" GT vs no-GT gap: {ca_gt_adp - ca_nogt_adp:.4f}")

 if ca_nogt_adp > baseline_adp:
 pct = (ca_nogt_adp - baseline_adp) / max(baseline_adp, 0.001) * 100
 log(f" >>> No-GT CA improves ADP by {pct:.0f}% — method is deployable without ground truth. <<<")
 else:
 log(f" >>> WARNING: No-GT CA does not improve over baseline. <<<")

 return {
 "top_k": top_k,
 "bias": bias,
 "iterations": iteration_results,
 "baseline_adp": round(baseline_adp, 4),
 "ca_with_gt_adp": round(ca_gt_adp, 4),
 "ca_without_gt_adp": ca_nogt_adp,
 "baseline_p10": round(base_p10, 4),
 "ca_with_gt_p10": round(ca_gt_p10, 4),
 "ca_without_gt_p10": ca_nogt_p10,
 }

# =============================================================================
# Experiment 5: Cross-Model CI Transfer
# =============================================================================
def cross_model_ci_experiment(ci_sum, all_malicious, node_to_attacks, flash_dir, alpha=1.0):
 """Experiment 5: Apply Orthrus's CI to Flash's node scores.

 Tests whether causal importance (computed from edge-level Orthrus losses)
 can improve detection in a node-level model (Flash) without retraining.
 """
 csv_files = sorted(glob.glob(os.path.join(flash_dir, "*.csv")))
 if not csv_files:
 log(f" ERROR: No CSV files found in {flash_dir}")
 return None

 flash_scores = {}
 for f in csv_files:
 df_flash = pd.read_csv(f)
 if "node" not in df_flash.columns or "loss" not in df_flash.columns:
 log(f" ERROR: Flash CSV must have 'node' and 'loss' columns. Found: {list(df_flash.columns)}")
 return None
 for _, row in df_flash.iterrows():
 nid = int(row["node"])
 loss = float(row["loss"])
 if nid not in flash_scores or loss > flash_scores[nid]:
 flash_scores[nid] = loss

 log("")
 log("=" * 72)
 log("--- Experiment 5: Cross-Model CI Transfer ---")
 log(f" Source: Orthrus CI (ci_sum for {len(ci_sum)} nodes)")
 log(f" Target: Flash node scores ({len(flash_scores)} nodes)")

 common_nodes = set(flash_scores.keys()) & set(ci_sum.keys())
 flash_malicious = set(flash_scores.keys()) & all_malicious
 log(f" Overlap: {len(common_nodes)} nodes with both Flash score and ci_sum")
 log(f" Malicious nodes in Flash: {len(flash_malicious)}")

 nodes = list(flash_scores.keys())
 labels_list = [1 if n in all_malicious else 0 for n in nodes]
 labels_arr = np.array(labels_list)

 flash_adp = compute_adp([flash_scores[n] for n in nodes], nodes, node_to_attacks, labels_list)
 flash_p10 = compute_precision_at_k([flash_scores[n] for n in nodes], nodes, labels_arr, k=10)

 # Apply Orthrus CI to Flash scores
 ca_flash_scores = {}
 for n in nodes:
 cs = ci_sum.get(n, 0.0)
 ca_flash_scores[n] = flash_scores[n] * float(sigmoid_array(alpha * cs))

 ca_adp = compute_adp([ca_flash_scores[n] for n in nodes], nodes, node_to_attacks, labels_list)
 ca_p10 = compute_precision_at_k([ca_flash_scores[n] for n in nodes], nodes, labels_arr, k=10)

 results = {
 "flash_nodes": len(flash_scores),
 "flash_malicious": len(flash_malicious),
 "flash_baseline_adp": round(flash_adp, 4),
 "flash_ca_adp": round(ca_adp, 4),
 "flash_baseline_p10": round(flash_p10, 4),
 "flash_ca_p10": round(ca_p10, 4),
 }

 log(f"\n {'':25s} {'ADP':>10s} {'Prec@10':>10s}")
 log(f" {'Flash baseline':25s} {flash_adp:>10.4f} {flash_p10:>10.4f}")
 log(f" {'Flash + Orthrus CI':25s} {ca_adp:>10.4f} {ca_p10:>10.4f}")
 delta = ca_adp - flash_adp
 log(f"\n Cross-model CA improvement: {delta:+.4f}")
 if delta > 0:
 log(f" >>> SUCCESS: Orthrus CI improves Flash detection by {delta / max(flash_adp, 0.001) * 100:.0f}% <<<")
 else:
 log(f" >>> CI transfer does not improve Flash detection. <<<")

 return results

# =============================================================================
# Experiment 4: Edge Attribution Method Comparison
# =============================================================================
def compare_edge_attribution_methods(ci_scores, all_edges, all_malicious):
 """Experiment 4: Compare CI vs other edge attribution methods.

 Measures how well each method's top-K edges overlap with attack edges
 (edges touching malicious nodes).
 """
 import random

 malicious_set = set(all_malicious)

 # Define attack edges: edges where src or dst is malicious
 attack_edges = set()
 for e in all_edges:
 if e["src"] in malicious_set or e["dst"] in malicious_set:
 attack_edges.add((e["src"], e["dst"], e["t"]))
 total_attack = len(attack_edges)
 total_edges = len(all_edges)
 attack_ratio = total_attack / total_edges if total_edges else 0

 # Sort edges by each method
 ci_sorted = sorted(ci_scores.items(), key=lambda x: x[1], reverse=True)
 loss_sorted = sorted(all_edges, key=lambda e: e["loss"], reverse=True)
 random.seed(42)
 random.shuffle(loss_sorted) # reuse for random baseline
 # Random: just take random indices
 random_indices = list(range(total_edges))
 random.shuffle(random_indices)

 ks = [5, 10, 20, 50]
 results = {
 "total_edges": total_edges,
 "total_attack_edges": total_attack,
 "attack_edge_ratio": round(attack_ratio, 6),
 "methods": [],
 }

 log("")
 log("=" * 72)
 log("--- Experiment 4: Edge Attribution Comparison ---")
 log(f"Total edges: {total_edges:,}, attack edges: {total_attack:,} ({attack_ratio:.4%})")
 log(f"\n{'K':>5s} {'CI':>12s} {'Max Loss':>12s} {'Random':>12s}")
 log("-" * 45)

 for K in ks:
 # CI top-K
 ci_topk = set(k for k, _ in ci_sorted[:K])
 ci_hits = len(ci_topk & attack_edges)
 ci_prec = ci_hits / K

 # Loss top-K
 loss_topk = set((e["src"], e["dst"], e["t"]) for e in loss_sorted[:K])
 loss_hits = len(loss_topk & attack_edges)
 loss_prec = loss_hits / K

 # Random top-K
 rand_topk = set()
 for idx in random_indices[:K]:
 e = all_edges[idx]
 rand_topk.add((e["src"], e["dst"], e["t"]))
 rand_hits = len(rand_topk & attack_edges)
 rand_prec = rand_hits / K

 results["methods"].append({
 "K": K,
 "ci_precision": round(ci_prec, 4),
 "ci_hits": ci_hits,
 "loss_precision": round(loss_prec, 4),
 "loss_hits": loss_hits,
 "random_precision": round(rand_prec, 4),
 "random_hits": rand_hits,
 })

 log(f"{K:>5d} {ci_prec:>8.1%} ({ci_hits}) {loss_prec:>8.1%} ({loss_hits}) {rand_prec:>8.1%} ({rand_hits})")

 # Summary
 ci_p10 = results["methods"][1]["ci_precision"]
 loss_p10 = results["methods"][1]["loss_precision"]
 rand_p10 = results["methods"][1]["random_precision"]

 log(f"\n CI Precision@10: {ci_p10:.1%} vs Loss: {loss_p10:.1%} vs Random: {rand_p10:.1%}")
 if ci_p10 > loss_p10:
 log(f" >>> CI outperforms Loss by {ci_p10/max(loss_p10,0.001):.1f}× at K=10. <<<")
 else:
 log(f" >>> CI does not outperform Loss at K=10. <<<")

 return results

# =============================================================================
# Task B: Attack Critical Path
# =============================================================================
def datetime_to_ns(dt_str):
 """Convert 'YYYY-MM-DD HH:MM:SS' (US/Eastern) to nanosecond timestamp.

 The DARPA TC config uses US/Eastern times (matches datetime_to_ns_time_US
 in pidsmaker/utils/utils.py). Using UTC would be 4 hours off during EDT.
 """
 try:
 import pytz
 tz = pytz.timezone("US/Eastern")
 dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
 dt = tz.localize(dt)
 return int(dt.timestamp() * 10**9)
 except ImportError:
 # Fallback: UTC (will be ~4h off, but better than crashing)
 log("WARNING: pytz not installed, falling back to UTC (attack time window may be wrong)")
 dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
 return int(dt.timestamp() * 10**9)

def build_attack_subgraph(all_edges, malicious_ids, hop=2, tw_start_ns=None, tw_end_ns=None):
 """Filter edges to the k-hop neighbourhood of malicious nodes.

 If tw_start_ns/tw_end_ns are provided, only edges within the time window are
 considered — this drastically reduces the subgraph size for large datasets.
 """
 malicious_set = set(malicious_ids)

 # Step 1: filter by time window (biggest reduction)
 if tw_start_ns is not None and tw_end_ns is not None:
 candidate_edges = [
 e for e in all_edges
 if tw_start_ns <= e["t"] <= tw_end_ns
 ]
 else:
 candidate_edges = all_edges

 # Step 2: build adjacency from filtered edges
 adjacency = defaultdict(set)
 for e in candidate_edges:
 adjacency[e["src"]].add(e["dst"])
 adjacency[e["dst"]].add(e["src"])

 # Step 3: BFS for k-hop neighbourhood
 frontier = set(malicious_set)
 neighborhood = set(malicious_set)
 for _ in range(hop):
 next_frontier = set()
 for n in frontier:
 next_frontier.update(adjacency.get(n, set()))
 next_frontier -= neighborhood
 neighborhood.update(next_frontier)
 frontier = next_frontier

 # Step 4: filter edges to neighbourhood
 subgraph_edges = [
 e for e in candidate_edges
 if e["src"] in neighborhood or e["dst"] in neighborhood
 ]
 return subgraph_edges, neighborhood

def find_entry_node(subgraph_edges, malicious_ids, tw_start_ns, tw_end_ns, preferred_ids=None):
 """Find the attack entry node: malicious node in the earliest edge within the attack time window.

 If preferred_ids is provided, only edges touching those nodes are considered first.
 Falls back to all malicious_ids if preferred_ids yields no candidates.
 """
 search_ids = preferred_ids if preferred_ids else malicious_ids
 search_set = set(search_ids)

 in_window = [
 e for e in subgraph_edges
 if tw_start_ns <= e["t"] <= tw_end_ns
 and (e["src"] in search_set or e["dst"] in search_set)
 ]

 # Fallback to all malicious_ids if preferred yielded nothing
 if not in_window and preferred_ids:
 search_set = set(malicious_ids)
 in_window = [
 e for e in subgraph_edges
 if tw_start_ns <= e["t"] <= tw_end_ns
 and (e["src"] in search_set or e["dst"] in search_set)
 ]

 candidates = in_window if in_window else subgraph_edges
 if not candidates:
 return None

 earliest = min(candidates, key=lambda e: e["t"])
 if earliest["src"] in malicious_ids:
 return earliest["src"]
 return earliest["dst"]

def find_top_anomaly_node(node_scores, malicious_ids, preferred_ids=None):
 """The malicious node with the highest baseline anomaly score.

 If preferred_ids is provided, only those nodes are considered first.
 Falls back to all malicious_ids if preferred yields nothing.
 """
 search_ids = list(preferred_ids) if preferred_ids else list(malicious_ids)
 scored = [(node_scores.get(n, 0.0), n) for n in search_ids if n in node_scores]

 if not scored and preferred_ids:
 scored = [(node_scores.get(n, 0.0), n) for n in malicious_ids if n in node_scores]

 if not scored:
 return None
 scored.sort(reverse=True)
 return scored[0][1]

def _build_ci_graph(ci_scores, malicious_ids):
 """Build a directed graph from ALL edges touching malicious nodes.

 Includes both CI > 0 edges (high weight) and CI = 0 edges (provide
 connectivity). No anti-parallel removal or cycle elimination — the
 graph is used as-is with Dijkstra (handles cycles naturally).

 For each (src, dst) pair, only the edge with the highest CI is kept.
 Each edge gets a Dijkstra weight = 1/(1+CI): high-CI edges have low
 weight (preferred), zero-CI edges have weight 1 (neutral).
 """
 import networkx as nx

 malicious_set = set(malicious_ids)

 # Collect max CI per (src, dst) pair, for ALL edges touching malicious nodes
 best_ci = {} # (src, dst) -> (ci, t)
 for (src, dst, t), ci in ci_scores.items():
 if src not in malicious_set and dst not in malicious_set:
 continue
 pair = (src, dst)
 if pair not in best_ci or ci > best_ci[pair][0]:
 best_ci[pair] = (ci, t)

 G = nx.DiGraph()
 for (u, v), (ci, t) in best_ci.items():
 # Dijkstra weight: high CI -> low weight -> preferred
 w = 1.0 / (1.0 + ci) if ci > 0 else 1.0
 G.add_edge(u, v, ci=ci, t=t, weight=w)

 return G

def _make_dag_temporal(G):
 """Convert a directed graph to a DAG using temporal ordering.

 Processes edges in timestamp order. An edge (u, v, t) is added only if
 there is no existing path from v to u (which would create a cycle).
 Much faster than nx.simple_cycles for large graphs.
 """
 import networkx as nx

 edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2].get("t", 0))
 dag = nx.DiGraph()
 dag.add_nodes_from(G.nodes())

 for u, v, data in edges_sorted:
 # Check if adding u->v would create a cycle (i.e., v can already reach u)
 if not nx.has_path(dag, v, u):
 dag.add_edge(u, v, **data)

 return dag

def _longest_path_dag(G, source, target):
 """Longest (max-CI) path from *source* to *target* via topological DP."""
 import networkx as nx

 if source not in G or target not in G:
 return None
 if not nx.is_directed_acyclic_graph(G):
 G = _make_dag_temporal(G)

 topo = list(nx.topological_sort(G))
 dist = {n: -float("inf") for n in G.nodes()}
 dist[source] = 0.0
 pred = {n: None for n in G.nodes()}

 for u in topo:
 if dist[u] == -float("inf"):
 continue
 for v in G.successors(u):
 w = G[u][v].get("ci", 0.0)
 if dist[u] + w > dist[v]:
 dist[v] = dist[u] + w
 pred[v] = u

 if dist[target] == -float("inf"):
 return None

 path = [target]
 while path[-1] != source:
 p = pred[path[-1]]
 if p is None:
 return None
 path.append(p)
 path.reverse()
 return path

def _bfs_path(subgraph_edges, source, target, undirected=False):
 """Find any path from source to target using BFS on the full subgraph.

 Uses pandas for fast adjacency building. Returns the shortest path (by
 hop count) or None if no path exists.
 """
 from collections import deque

 if source == target:
 return [source]

 df = pd.DataFrame(subgraph_edges)

 if undirected:
 # Build undirected adjacency: both src->dst and dst->src
 fwd = df[["src", "dst"]]
 rev = df[["dst", "src"]].rename(columns={"dst": "src", "src": "dst"})
 undirected_df = pd.concat([fwd, rev], ignore_index=True)
 adj = undirected_df.groupby("src")["dst"].apply(list).to_dict()
 else:
 adj = df.groupby("src")["dst"].apply(list).to_dict()

 queue = deque([source])
 visited = {source}
 pred = {source: None}

 while queue:
 u = queue.popleft()
 for v in adj.get(u, []):
 if v not in visited:
 visited.add(v)
 pred[v] = u
 if v == target:
 path = [target]
 while path[-1] != source:
 path.append(pred[path[-1]])
 path.reverse()
 return path
 queue.append(v)

 return None

def _top_k_ci_path(ci_scores, malicious_ids, k=10):
 """Build a path from the top-K highest-CI edges touching malicious nodes.

 Returns a list of nodes forming the best chain, or the top edge if no
 chain can be formed.
 """
 malicious_set = set(malicious_ids)

 # Collect CI > 0 edges touching malicious nodes
 ci_edges = []
 for (src, dst, t), ci in ci_scores.items():
 if ci > 0 and (src in malicious_set or dst in malicious_set):
 ci_edges.append((ci, src, dst))
 ci_edges.sort(reverse=True)

 if not ci_edges:
 return []

 # Greedily connect edges: start from the highest CI edge, extend if possible
 _, s, d = ci_edges[0]
 path = [s, d]
 used = {(s, d)}

 for ci, src, dst in ci_edges[1:]:
 # Try to prepend: src matches path[0]
 if dst == path[0] and (src, dst) not in used:
 path.insert(0, src)
 used.add((src, dst))
 # Try to append: dst matches path[-1]
 elif src == path[-1] and (src, dst) not in used:
 path.append(dst)
 used.add((src, dst))

 return path

def find_critical_path(subgraph_edges, ci_scores, node_scores, malicious_ids, tw_start_ns, tw_end_ns, preferred_ids=None):
 """Find the attack critical path in the malicious-node subgraph.

 Two-tier approach:
 1. Dijkstra on CI-weighted graph (handles cycles, no DAG conversion).
 2. If no path (graph disconnected), BFS fallback on the full subgraph.

 If preferred_ids is provided, entry/top nodes are selected from attack-specific
 nodes first (excludes nodes shared across multiple attacks).
 """
 entry = find_entry_node(subgraph_edges, malicious_ids, tw_start_ns, tw_end_ns, preferred_ids)
 top_node = find_top_anomaly_node(node_scores, malicious_ids, preferred_ids)

 if entry is None or top_node is None:
 return {"path": [], "entry": entry, "top_anomaly": top_node, "total_ci": 0.0}

 if entry == top_node:
 return {"path": [entry], "entry": entry, "top_anomaly": top_node, "total_ci": 0.0}

 # --- Tier 1: Dijkstra on CI-weighted graph (handles cycles, no DAG conversion) ---
 G = _build_ci_graph(ci_scores, malicious_ids)
 log(f" CI graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

 G.add_node(entry)
 G.add_node(top_node)

 path = None
 try:
 import networkx as nx
 path = nx.dijkstra_path(G, entry, top_node, weight="weight")
 except (nx.NetworkXNoPath, nx.NodeNotFound):
 pass

 if path is not None:
 total_ci = 0.0
 for i in range(len(path) - 1):
 if G.has_edge(path[i], path[i + 1]):
 total_ci += G[path[i]][path[i + 1]].get("ci", 0.0)
 return {"path": path, "entry": entry, "top_anomaly": top_node, "total_ci": total_ci}

 # --- Tier 2: Directed BFS fallback on full subgraph ---
 log(f" CI graph disconnected ({entry} -> {top_node}), falling back to BFS...")
 path = _bfs_path(subgraph_edges, entry, top_node)

 # --- Tier 2.5: Undirected BFS (provenance edges may be reversed) ---
 if path is None:
 log(f" Directed BFS failed, trying undirected BFS...")
 path = _bfs_path(subgraph_edges, entry, top_node, undirected=True)

 # --- Tier 3: Top-K CI edges (no path between entry and top) ---
 if path is None:
 log(f" No path found between {entry} and {top_node}.")
 log(f" Reporting top-K CI edges as critical path...")
 path = _top_k_ci_path(ci_scores, malicious_ids)
 if len(path) >= 2:
 log(f" Top-K CI path: {len(path)} nodes from highest-CI edges")
 else:
 log(f" No CI > 0 edges found for this attack.")
 return {"path": [], "entry": entry, "top_anomaly": top_node, "total_ci": 0.0}

 # Compute total CI along the BFS path (lookup from non-zero CI edges only)
 ci_lookup = {}
 for (s, d, _), ci in ci_scores.items():
 if ci > 0.0:
 pair = (s, d)
 if pair not in ci_lookup or ci > ci_lookup[pair]:
 ci_lookup[pair] = ci

 total_ci = 0.0
 for i in range(len(path) - 1):
 total_ci += ci_lookup.get((path[i], path[i + 1]), 0.0)

 log(f" BFS path found ({len(path)} nodes, {len(path)-1} edges, CI={total_ci:.4f})")
 return {"path": path, "entry": entry, "top_anomaly": top_node, "total_ci": total_ci}

def compute_acp_iou(path, subgraph_edges, malicious_ids):
 """ACP-IoU: IoU between identified path edges and ground-truth edges.

 ground_truth_edges = {(src, dst) for edge in subgraph if src or dst is malicious}
 identified_path_edges = {(path[i], path[i+1]) for i in range(len(path)-1)}
 IoU = |intersection| / |union|
 """
 malicious_set = set(malicious_ids)

 gt_pairs = set()
 for e in subgraph_edges:
 if e["src"] in malicious_set or e["dst"] in malicious_set:
 gt_pairs.add((e["src"], e["dst"]))

 path_pairs = set()
 for i in range(len(path) - 1):
 path_pairs.add((path[i], path[i + 1]))

 if not gt_pairs and not path_pairs:
 return 0.0, 0, 0, 0

 intersection = gt_pairs & path_pairs
 union = gt_pairs | path_pairs
 iou = len(intersection) / len(union) if union else 0.0
 return iou, len(intersection), len(path_pairs), len(gt_pairs)

def format_node_label(node_id, node_labels):
 """Short human-readable label for a node ID.

 Examples:
 {'subject': 'None nginx'} -> nginx(211776)
 {'netflow': '128.55...'} -> netflow(26277)
 {'file': '/tmp/vUgefal'} -> vUgefal(1219519)
 {'file': 'None'} -> file(379482)
 """
 import ast

 lbl = node_labels.get(node_id, "")
 if isinstance(lbl, str) and lbl:
 try:
 d = ast.literal_eval(lbl)
 if isinstance(d, dict):
 for k, v in d.items():
 if not v or v == "None":
 return f"{k}({node_id})"
 if k == "subject" and isinstance(v, str):
 # "None nginx" -> "nginx"
 parts = v.split()
 return f"{parts[-1] if parts else v}({node_id})"
 if k == "file" and isinstance(v, str):
 # "/tmp/vUgefal" -> "vUgefal"
 name = os.path.basename(v) if "/" in v else v
 return f"{name}({node_id})"
 if k == "netflow":
 return f"netflow({node_id})"
 return f"{v}({node_id})"
 except Exception:
 pass
 return str(node_id)

def run_task_b(all_edges, ci_scores, node_scores, attack_to_malicious, node_labels, attacks_info):
 """Run Attack Critical Path identification for each attack scenario.

 Args:
 attacks_info: dataset-specific ATTACKS dict with tw_start/tw_end per attack
 """
 results = {}
 log("")
 log("--- Task B: Attack Critical Path ---")

 # Compute shared nodes (appearing in multiple attacks) to prioritize attack-specific nodes
 node_attack_count = defaultdict(int)
 for ids in attack_to_malicious.values():
 for nid in ids:
 node_attack_count[nid] += 1
 shared_ids = {nid for nid, count in node_attack_count.items() if count > 1}
 if shared_ids:
 log(f" Cross-attack shared nodes: {len(shared_ids)} (will prefer attack-specific nodes)")

 for attack_name, malicious_ids in attack_to_malicious.items():
 info = attacks_info[attack_name]
 tw_start = datetime_to_ns(info["tw_start"])
 tw_end = datetime_to_ns(info["tw_end"])

 # Attack-specific nodes (not shared with other attacks)
 preferred_ids = malicious_ids - shared_ids
 if preferred_ids:
 log(f"\n{attack_name}: ({len(preferred_ids)} attack-specific, {len(malicious_ids) - len(preferred_ids)} shared)")
 else:
 log(f"\n{attack_name}:")

 subgraph_edges, neighborhood = build_attack_subgraph(
 all_edges, malicious_ids, hop=2
 )
 log(f" Subgraph: {len(neighborhood)} nodes, {len(subgraph_edges)} edges (2-hop)")

 if not subgraph_edges:
 log(" ERROR: empty subgraph, skipping.")
 results[attack_name] = {"path": [], "acp_iou": 0.0}
 continue

 cp = find_critical_path(
 subgraph_edges, ci_scores, node_scores, malicious_ids, tw_start, tw_end,
 preferred_ids=preferred_ids if preferred_ids else None,
 )
 path = cp["path"]

 if not path:
 log(" No critical path found.")
 results[attack_name] = {
 "path": [],
 "path_length": 0,
 "entry": cp.get("entry"),
 "top_anomaly": cp.get("top_anomaly"),
 "total_ci": 0.0,
 "acp_iou": 0.0,
 }
 continue

 # Format path for display
 path_labels = [format_node_label(n, node_labels) for n in path]
 log(f" Entry node: {format_node_label(cp['entry'], node_labels)}")
 log(f" Top anomaly node: {format_node_label(cp['top_anomaly'], node_labels)}")
 log(f" Path ({len(path)} nodes, {len(path)-1} edges):")
 log(f" {' -> '.join(path_labels)}")
 log(f" Total CI along path: {cp['total_ci']:.4f}")

 iou, inter, path_n, gt_n = compute_acp_iou(path, subgraph_edges, malicious_ids)
 log(f" ACP-IoU: {iou:.4f} (intersection={inter}, path_edges={path_n}, gt_edges={gt_n})")

 results[attack_name] = {
 "path": path,
 "path_length": len(path) - 1,
 "entry": cp["entry"],
 "top_anomaly": cp["top_anomaly"],
 "total_ci": round(cp["total_ci"], 4),
 "acp_iou": round(iou, 4),
 "gt_edges": gt_n,
 "path_edges_in_gt": inter,
 }

 avg_iou = np.mean([r.get("acp_iou", 0.0) for r in results.values()]) if results else 0.0
 log(f"\n Average ACP-IoU: {avg_iou:.4f}")
 return results, avg_iou

# =============================================================================
# Main
# =============================================================================
def main():
 log("=" * 72)
 log("Stage 3: Causal-Aware Detection + Attack Critical Path")
 log("=" * 72)

 # --- 1. Parse config ---
 args = get_runtime_required_args()
 cfg = get_yml_cfg(args)
 dataset_name = cfg.dataset.name
 log(f"Dataset: {dataset_name}")

 # Parse --epoch and --alpha from argv (get_runtime_required_args may not expose them)
 specified_epoch = None
 alpha = 1.0
 run_exp1 = False
 run_exp2 = False
 run_exp3 = False
 run_exp4 = False
 cross_model_dir = None
 losses_dir_override = None
 for i, arg in enumerate(sys.argv):
 if arg == "--epoch" and i + 1 < len(sys.argv):
 specified_epoch = int(sys.argv[i + 1])
 elif arg == "--alpha" and i + 1 < len(sys.argv):
 alpha = float(sys.argv[i + 1])
 elif arg == "--exp1":
 run_exp1 = True
 elif arg == "--exp2":
 run_exp2 = True
 elif arg == "--exp3":
 run_exp3 = True
 elif arg == "--exp4":
 run_exp4 = True
 elif arg == "--cross-model-dir" and i + 1 < len(sys.argv):
 cross_model_dir = sys.argv[i + 1]
 elif arg == "--losses-dir" and i + 1 < len(sys.argv):
 losses_dir_override = sys.argv[i + 1]
 log(f"Alpha (sigmoid scaling): {alpha}")
 if run_exp1:
 log("Experiment 1 (Causal Necessity) enabled")
 if run_exp2:
 log("Experiment 2 (False Positive Analysis) enabled")
 if run_exp3:
 log("Experiment 3 (No Ground Truth Deployment) enabled")
 if run_exp4:
 log("Experiment 4 (Edge Attribution Comparison) enabled")
 if cross_model_dir:
 log(f"Experiment 5 (Cross-Model CI Transfer) enabled: {cross_model_dir}")

 # --- 2. Load ground truth ---
 log("\n--- Loading ground truth ---")
 attack_to_malicious, node_labels, node_to_attacks = load_ground_truth(dataset_name)
 all_malicious = set()
 for ids in attack_to_malicious.values():
 all_malicious.update(ids)
 log(f"Total unique malicious nodes across attacks: {len(all_malicious)}")

 # --- 3. Load edge losses ---
 log("\n--- Finding edge losses ---")
 if losses_dir_override:
 edge_losses_dir = losses_dir_override
 log(f"Using specified edge losses dir: {edge_losses_dir}")
 else:
 edge_losses_dir = find_edge_losses_dir(cfg)
 log(f"Edge losses dir: {edge_losses_dir}")
 epochs = list_available_epochs(edge_losses_dir)
 log(f"Available epochs: {epochs}")

 all_edges, used_epoch = load_edge_losses(edge_losses_dir, specified_epoch)
 if not all_edges:
 log("ERROR: No edges loaded. Check that training produced edge_losses CSVs.")
 sys.exit(1)

 # Confirm malicious nodes are present
 test_nodes = set()
 for e in all_edges:
 test_nodes.add(e["src"])
 test_nodes.add(e["dst"])
 found = all_malicious & test_nodes
 log(f"\nMalicious nodes present in test set: {len(found)}/{len(all_malicious)}")
 for aname, ids in attack_to_malicious.items():
 f = ids & test_nodes
 log(f" {aname}: {len(f)}/{len(ids)} malicious nodes in test set")

 # --- 4. Compute CI ---
 log("\n--- Computing Causal Importance (CI) ---")
 use_dst = True # matches orthrus.yml use_dst_node_loss: True
 ci_scores, node_scores = compute_ci_scores(
 all_edges, all_malicious, use_dst_loss=use_dst
 )

 ci_sum = compute_ci_sum_per_node(ci_scores)
 log(f"Nodes with non-zero ci_sum: {len(ci_sum)}")
 if ci_sum:
 log(f"ci_sum range: [{min(ci_sum.values()):.4f}, {max(ci_sum.values()):.4f}]")

 # --- 5. Compute CA scores ---
 log("\n--- Computing Causal-Aware (CA) scores ---")
 ca_scores = compute_ca_scores(node_scores, ci_sum, alpha)

 # Show CA effect on malicious vs benign
 mal_baseline = [node_scores[n] for n in all_malicious if n in node_scores]
 mal_ca = [ca_scores[n] for n in all_malicious if n in ca_scores]
 benign_sample = [n for n in list(node_scores.keys())[:5000] if n not in all_malicious]
 ben_baseline = [node_scores[n] for n in benign_sample]
 ben_ca = [ca_scores[n] for n in benign_sample]
 if mal_baseline:
 log(f"Malicious nodes: baseline avg={np.mean(mal_baseline):.4f}, CA avg={np.mean(mal_ca):.4f}, "
 f"ratio={np.mean(mal_ca)/max(np.mean(mal_baseline),1e-9):.3f}")
 if ben_baseline:
 log(f"Benign (sample): baseline avg={np.mean(ben_baseline):.4f}, CA avg={np.mean(ben_ca):.4f}, "
 f"ratio={np.mean(ben_ca)/max(np.mean(ben_baseline),1e-9):.3f}")

 # --- 6. Task A ---
 task_a_results = run_task_a(node_scores, ca_scores, node_to_attacks, attack_to_malicious)

 # --- 6.5 Experiment 1: Causal Necessity (optional) ---
 exp1_results = None
 if run_exp1:
 exp1_results = causal_necessity_experiment(
 all_edges, ci_scores, node_scores, node_to_attacks, all_malicious, use_dst_loss=use_dst
 )

 # --- 6.6 Experiment 2: False Positive Analysis (optional) ---
 exp2_results = None
 if run_exp2:
 exp2_results = false_positive_analysis(
 node_scores, ca_scores, ci_sum, all_malicious
 )

 # --- 6.7 Experiment 3: No Ground Truth Deployment (optional) ---
 exp3_results = None
 if run_exp3:
 exp3_results = no_gt_deployment_experiment(
 all_edges, node_scores, node_to_attacks, all_malicious,
 ca_scores, use_dst_loss=use_dst
 )

 # --- 6.8 Experiment 4: Edge Attribution Comparison (optional) ---
 exp4_results = None
 if run_exp4:
 exp4_results = compare_edge_attribution_methods(
 ci_scores, all_edges, all_malicious
 )

 # --- 6.9 Experiment 5: Cross-Model CI Transfer (optional) ---
 exp5_results = None
 if cross_model_dir:
 exp5_results = cross_model_ci_experiment(
 ci_sum, all_malicious, node_to_attacks, cross_model_dir, alpha=alpha
 )

 # --- 7. Task B ---
 task_b_results, avg_iou = run_task_b(
 all_edges, ci_scores, node_scores, attack_to_malicious, node_labels,
 DATASET_ATTACKS[dataset_name],
 )

 # --- 8. Summary ---
 log("\n" + "=" * 72)
 log("--- Summary ---")
 log(f"{'Method':<18s} {'ADP':>8s} {'Prec@10':>10s} {'ACP-IoU(avg)':>14s}")
 log("-" * 52)
 b = task_a_results["baseline"]
 c = task_a_results["causal_aware"]
 log(f"{'Baseline':<18s} {b['adp']:>8.4f} {b['precision_at_10']:>10.4f} {'N/A':>14s}")
 log(f"{'Causal-Aware':<18s} {c['adp']:>8.4f} {c['precision_at_10']:>10.4f} {avg_iou:>14.4f}")

 imp = task_a_results["improvement"]
 log(f"\nADP improvement: {imp['adp_delta']:+.4f}")
 log(f"Precision@10 improvement: {imp['precision_delta']:+.4f}")
 log(f"Average ACP-IoU: {avg_iou:.4f}")

 verdict = "PASS" if (imp["adp_delta"] >= 0 and avg_iou > 0) else "REVIEW"
 log(f"\nVerdict: {verdict}")
 if imp["adp_delta"] < 0:
 log(" (ADP degraded - try --alpha 0.5 or --alpha 2.0)")
 if avg_iou == 0:
 log(" (ACP-IoU is 0 - path may not overlap ground truth, check Task B output)")

 # --- 9. Save JSON ---
 out_dir = cfg._artifact_dir
 os.makedirs(out_dir, exist_ok=True)
 out_file = os.path.join(out_dir, "stage3_results.json")
 full_results = {
 "stage": 3,
 "dataset": dataset_name,
 "epoch": used_epoch,
 "alpha": alpha,
 "task_a": task_a_results,
 "task_b": task_b_results,
 "experiment_1": exp1_results,
 "experiment_2": exp2_results,
 "experiment_3": exp3_results,
 "experiment_4": exp4_results,
 "experiment_5": exp5_results,
 "summary": {
 "baseline_adp": b["adp"],
 "causal_aware_adp": c["adp"],
 "adp_delta": imp["adp_delta"],
 "average_acp_iou": round(avg_iou, 4),
 "verdict": verdict,
 },
 }
 with open(out_file, "w") as f:
 json.dump(full_results, f, indent=2, default=str)
 log(f"\nResults saved to {out_file}")

 log("\n" + "=" * 72)
 log("Stage 3 pipeline finished.")
 log("=" * 72)

if __name__ == "__main__":
 main()
