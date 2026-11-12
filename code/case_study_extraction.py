#!/usr/bin/env python3
"""
Extract Backdoor_06 attack subgraph data for case study visualization.

Self-contained: does NOT import pidsmaker.config (which differs across installs).
You MUST pass --losses-dir explicitly (same path you pass to stage3_causal_pipeline.py).

Usage (on server):
 cd ${WORKSPACE:-.}/code/PIDSMaker
 python case_study_extraction.py orthrus CADETS_E3 --epoch 13 \\
 --losses-dir ${WORKSPACE:-.}/artifacts/training/training/<HASH>/CADETS_E3/edge_losses/test \\
 --attack Backdoor_06

Outputs four files under results/case_study_<attack>_<dataset>_<model>_ep<N>/:
 - nodes.csv : node_id, label, baseline_score, ca_score, ci_sum, is_malicious, attack_role
 - edges.csv : src, dst, edge_type, loss, ci_score, ci_rank
 - subgraph.json : Graphviz / D3 / ECharts friendly
 - ranking.json : top-20 baseline vs top-20 CA
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# Only reuse stage3_causal_pipeline's CI/CA/ground-truth helpers
sys.path.insert(0, str(Path(__file__).parent))
from stage3_causal_pipeline import (
 load_ground_truth,
 load_edge_losses,
 compute_ci_scores,
 compute_ci_sum_per_node,
 compute_ca_scores,
)

def main():
 p = argparse.ArgumentParser(
 description=__doc__,
 formatter_class=argparse.RawDescriptionHelpFormatter,
 )
 p.add_argument("model", choices=["orthrus", "velox", "kairos"],
 help="PIDS architecture (only used to label output files)")
 p.add_argument("dataset", choices=["CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3"])
 p.add_argument("--epoch", type=int, required=True)
 p.add_argument("--losses-dir", required=True,
 help="Path to edge_losses/test directory (same as stage3 --losses-dir)")
 p.add_argument("--attack", default="Backdoor_06")
 p.add_argument("--alpha", type=float, default=1.0)
 p.add_argument("--top-fp", type=int, default=20)
 args = p.parse_args()

 losses_dir = args.losses_dir
 if not Path(losses_dir).is_dir():
 print(f"ERROR: --losses-dir '{losses_dir}' does not exist or is not a directory")
 sys.exit(1)

 print(f"Dataset: {args.dataset}")
 print(f"Attack: {args.attack}")
 print(f"Losses: {losses_dir}")
 print(f"Epoch: {args.epoch}")

 # ---- Ground truth (no cfg needed) ----
 attack_to_malicious, node_labels, node_to_attacks = load_ground_truth(args.dataset)
 if args.attack not in attack_to_malicious:
 print(f"ERROR: attack '{args.attack}' not in dataset. "
 f"Available: {list(attack_to_malicious.keys())}")
 sys.exit(1)

 malicious_for_attack = attack_to_malicious[args.attack]
 all_malicious = set()
 for ids in attack_to_malicious.values():
 all_malicious.update(ids)
 print(f"Malicious nodes for {args.attack}: {len(malicious_for_attack)}")
 print(f"Total malicious in dataset: {len(all_malicious)}")

 # ---- Load edges ----
 all_edges, used_epoch = load_edge_losses(losses_dir, args.epoch)
 print(f"Loaded {len(all_edges)} edges from epoch {used_epoch}")

 # ---- Compute CI / CA ----
 ci_scores, node_scores = compute_ci_scores(all_edges, all_malicious, use_dst_loss=True)
 ci_sum = compute_ci_sum_per_node(ci_scores)
 ca_scores = compute_ca_scores(node_scores, ci_sum, alpha=args.alpha, bias=0.0)
 print(f"Nodes with non-zero ci_sum: {len(ci_sum)}")

 # ---- Build attack subgraph (1-hop expansion around malicious nodes) ----
 mal_set = set(malicious_for_attack)
 sub_edges = [e for e in all_edges if e["src"] in mal_set or e["dst"] in mal_set]
 sub_nodes = set(mal_set)
 for e in sub_edges:
 sub_nodes.add(e["src"])
 sub_nodes.add(e["dst"])

 ci_lookup = {(e["src"], e["dst"], e["t"]):
 ci_scores.get((e["src"], e["dst"], e["t"]), 0.0)
 for e in sub_edges}

 # ---- nodes.csv ----
 nodes_data = []
 for n in sub_nodes:
 is_mal = n in mal_set
 nodes_data.append({
 "node_id": n,
 "label": node_labels.get(n, f"node_{n}"),
 "baseline_score": round(node_scores.get(n, 0.0), 4),
 "ca_score": round(ca_scores.get(n, 0.0), 4),
 "ci_sum": round(ci_sum.get(n, 0.0), 4),
 "is_malicious": int(is_mal),
 "attack_role": (args.attack if is_mal
 else (",".join(sorted(node_to_attacks.get(n, set())))
 if n in all_malicious else "benign")),
 })
 nodes_data.sort(key=lambda r: r["baseline_score"], reverse=True)

 # ---- edges.csv ----
 edges_data = []
 ci_ranked = sorted(
 sub_edges,
 key=lambda e: ci_lookup.get((e["src"], e["dst"], e["t"]), 0.0),
 reverse=True,
 )
 for rank, e in enumerate(ci_ranked):
 edges_data.append({
 "src": e["src"],
 "dst": e["dst"],
 "edge_type": e.get("edge_type", ""),
 "loss": round(float(e.get("loss", 0.0)), 4),
 "ci_score": round(ci_lookup.get((e["src"], e["dst"], e["t"]), 0.0), 4),
 "ci_rank": rank,
 "src_is_malicious": int(e["src"] in mal_set),
 "dst_is_malicious": int(e["dst"] in mal_set),
 })

 # ---- ranking.json ----
 all_nodes = list(node_scores.keys())
 top_baseline = sorted(all_nodes, key=lambda n: node_scores[n], reverse=True)[:args.top_fp]
 top_ca = sorted(all_nodes, key=lambda n: ca_scores.get(n, 0.0), reverse=True)[:args.top_fp]

 def describe(n):
 return {
 "node_id": n,
 "label": node_labels.get(n, f"node_{n}"),
 "baseline_score": round(node_scores.get(n, 0.0), 4),
 "ca_score": round(ca_scores.get(n, 0.0), 4),
 "ci_sum": round(ci_sum.get(n, 0.0), 4),
 "is_malicious": int(n in all_malicious),
 "attack": ",".join(sorted(node_to_attacks.get(n, set())))
 if n in all_malicious else "benign",
 }

 ranking = {
 "top_baseline": [describe(n) for n in top_baseline],
 "top_ca": [describe(n) for n in top_ca],
 "n_baseline_malicious_in_top": sum(1 for n in top_baseline if n in all_malicious),
 "n_ca_malicious_in_top": sum(1 for n in top_ca if n in all_malicious),
 }

 # ---- Write outputs ----
 out_dir = Path(__file__).parent.parent / "results" / f"case_study_{args.attack}_{args.dataset}_{args.model}_ep{args.epoch}"
 out_dir.mkdir(parents=True, exist_ok=True)

 with (out_dir / "nodes.csv").open("w", newline="") as f:
 w = csv.DictWriter(f, fieldnames=list(nodes_data[0].keys()))
 w.writeheader()
 w.writerows(nodes_data)

 with (out_dir / "edges.csv").open("w", newline="") as f:
 w = csv.DictWriter(f, fieldnames=list(edges_data[0].keys()))
 w.writeheader()
 w.writerows(edges_data)

 with (out_dir / "ranking.json").open("w") as f:
 json.dump(ranking, f, indent=2)

 subgraph_json = {
 "attack": args.attack,
 "dataset": args.dataset,
 "model": args.model,
 "epoch": args.epoch,
 "n_malicious": len(mal_set),
 "n_subgraph_nodes": len(sub_nodes),
 "n_subgraph_edges": len(sub_edges),
 "nodes": nodes_data,
 "edges": [{"src": e["src"], "dst": e["dst"],
 "ci": ci_lookup.get((e["src"], e["dst"], e["t"]), 0.0),
 "loss": e.get("loss", 0.0)} for e in sub_edges],
 }
 with (out_dir / "subgraph.json").open("w") as f:
 json.dump(subgraph_json, f, indent=2)

 print()
 print(f"=== Case study data written to: {out_dir}/ ===")
 print(f" {args.attack} malicious nodes: {len(mal_set)}")
 print(f" Subgraph (1-hop): {len(sub_nodes)} nodes, {len(sub_edges)} edges")
 print(f" Top-{args.top_fp} baseline contains malicious: {ranking['n_baseline_malicious_in_top']}")
 print(f" Top-{args.top_fp} CA contains malicious: {ranking['n_ca_malicious_in_top']}")
 print()
 print(f" Render subgraph.json online at:")
 print(f" https://dreampuf.github.io/GraphvizOnline/")

if __name__ == "__main__":
 main()
