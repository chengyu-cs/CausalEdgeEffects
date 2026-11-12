#!/usr/bin/env python3
"""
Experiment B: Reference Set Sensitivity Analysis

Tests how CA performance varies with |M| (number of known malicious endpoints).
Answers reviewer R1's question: "Does CA still work with |M|=2? |M|=1?"

Usage (on server):
 cd ${WORKSPACE:-.}/code/PIDSMaker
 python experiments/reference_set_size_sensitivity.py orthrus CADETS_E3 --epoch 13 \
 --losses-dir ${WORKSPACE:-.}/artifacts/training/training/<HASH>/CADETS_E3/edge_losses/test

 # Run for all 3 datasets:
 for ds in CADETS_E3 THEIA_E3 CLEARSCOPE_E3; do
 python experiments/reference_set_size_sensitivity.py orthrus $ds --epoch 13 \
 --losses-dir <HASH_PATH>/$ds/edge_losses/test
 done

Output:
 results/exp_b_sensitivity_<dataset>_ep<N>.json
 results/exp_b_sensitivity_<dataset>_ep<N>.png (plot)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage3_causal_pipeline import (
 load_ground_truth,
 load_edge_losses,
 compute_ci_scores,
 compute_ci_sum_per_node,
 compute_ca_scores,
 compute_adp,
)

def run_sensitivity(all_edges, node_scores, all_malicious, node_to_attacks, alpha=1.0,
 n_trials=5):
 """Vary |M| from 1 to |all_malicious|, measure CA ADP."""
 import random
 full_M = sorted(all_malicious)
 n_full = len(full_M)

 # Test sizes: 1, 2, 5, 10, n//4, n//2, n
 sizes = sorted(set([1, 2, 5, 10, n_full // 4, n_full // 2, n_full]))
 sizes = [s for s in sizes if 0 < s <= n_full]

 results = {}
 all_nodes = list(node_scores.keys())
 baseline_scores = list(node_scores.values())
 labels = [1 if n in all_malicious else 0 for n in all_nodes]
 baseline_adp = compute_adp(
 baseline_scores, all_nodes, node_to_attacks, labels
 )
 results["baseline_adp"] = round(baseline_adp, 4)
 results["full_M_size"] = n_full

 print(f"\n{'|M|':>8s} {'Mean ADP':>10s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
 print("-" * 50)

 for size in sizes:
 trial_adps = []
 for trial in range(n_trials):
 random.seed(trial * 100 + size)
 if size < n_full:
 M = set(random.sample(full_M, size))
 else:
 M = set(full_M)

 ci_scores, _ = compute_ci_scores(all_edges, M, use_dst_loss=True)
 ci_sum = compute_ci_sum_per_node(ci_scores)
 ca_scores = compute_ca_scores(node_scores, ci_sum, alpha=alpha, bias=0.0)
 ca_list = [ca_scores.get(n, 0.0) for n in all_nodes]

 adp = compute_adp(ca_list, all_nodes, node_to_attacks, labels)
 trial_adps.append(adp)

 results[size] = {
 "mean_adp": round(float(np.mean(trial_adps)), 4),
 "std_adp": round(float(np.std(trial_adps)), 4),
 "min_adp": round(float(np.min(trial_adps)), 4),
 "max_adp": round(float(np.max(trial_adps)), 4),
 "trials": [round(a, 4) for a in trial_adps],
 }
 print(f"{size:>8d} {np.mean(trial_adps):>10.4f} {np.std(trial_adps):>8.4f} "
 f"{np.min(trial_adps):>8.4f} {np.max(trial_adps):>8.4f}")

 return results

def plot_results(results, dataset, out_path):
 """Plot ADP vs |M| curve."""
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt

 sizes = sorted(k for k in results if isinstance(k, int))
 means = [results[s]["mean_adp"] for s in sizes]
 stds = [results[s]["std_adp"] for s in sizes]
 baseline = results["baseline_adp"]

 fig, ax = plt.subplots(figsize=(4, 3))
 ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, color="#D55E00", lw=1.5)
 ax.axhline(y=baseline, color="#999999", ls="--", label=f"Baseline ADP={baseline:.3f}")
 ax.axhline(y=results[sizes[-1]]["mean_adp"], color="#009E73", ls=":",
 label=f"Full |M| ADP={results[sizes[-1]]['mean_adp']:.3f}")
 ax.set_xlabel("|M| (reference set size)")
 ax.set_ylabel("CA ADP")
 ax.set_title(f"{dataset}: Reference Set Sensitivity")
 ax.legend(fontsize=7)
 ax.set_xscale("log")
 fig.tight_layout()
 fig.savefig(out_path, dpi=200)
 print(f"Plot saved: {out_path}")

def main():
 p = argparse.ArgumentParser(description=__doc__,
 formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument("model", choices=["orthrus", "velox", "kairos"])
 p.add_argument("dataset", choices=["CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3"])
 p.add_argument("--epoch", type=int, required=True)
 p.add_argument("--losses-dir", required=True)
 p.add_argument("--alpha", type=float, default=1.0)
 p.add_argument("--n-trials", type=int, default=5)
 args = p.parse_args()

 print(f"Dataset: {args.dataset}, Model: {args.model}, Epoch: {args.epoch}")

 attack_to_malicious, node_labels, node_to_attacks = load_ground_truth(args.dataset)
 all_malicious = set()
 for ids in attack_to_malicious.values():
 all_malicious.update(ids)
 print(f"Total malicious nodes: {len(all_malicious)}")

 all_edges, used_epoch = load_edge_losses(args.losses_dir, args.epoch)
 print(f"Loaded {len(all_edges)} edges from epoch {used_epoch}")

 ci_scores, node_scores = compute_ci_scores(all_edges, all_malicious, use_dst_loss=True)
 print(f"Nodes with non-zero ci_sum: {len(compute_ci_sum_per_node(ci_scores))}")

 results = run_sensitivity(
 all_edges, node_scores, all_malicious, node_to_attacks,
 alpha=args.alpha, n_trials=args.n_trials,
 )

 out_dir = Path(__file__).parent.parent.parent / "results"
 out_dir.mkdir(exist_ok=True)
 json_path = out_dir / f"exp_b_sensitivity_{args.dataset}_ep{used_epoch}.json"
 json_path.write_text(json.dumps(results, indent=2))
 print(f"\nResults saved: {json_path}")

 plot_path = out_dir / f"exp_b_sensitivity_{args.dataset}_ep{used_epoch}.png"
 try:
 plot_results(results, args.dataset, plot_path)
 except Exception as e:
 print(f"Plot failed (non-critical): {e}")

if __name__ == "__main__":
 main()
