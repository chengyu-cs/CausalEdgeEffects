"""Causal Screening Prototype for PIDSMaker/Orthrus — CSV-based version.

This script implements Stage 2 of the Causal Provenance research plan:
validating that Causal Importance (CI) scores can identify attack edges.

It reads pre-computed edge losses from the training artifacts (no model loading
or forward pass needed), then performs leave-one-out edge interventions on the
Backdoor_06 attack subgraph to compute ICE (Individual Causal Effect).

Because Orthrus uses use_memory=False and insert_neighbors_before=False, removing
an edge from a batch only affects the loss aggregation. The ICE formula reduces to:

 ICE(e, v) = max(0, score(v) - max(losses of v's other edges))

This is exact (not an approximation) for this model configuration.

Usage:
 python causal_screening_prototype.py orthrus CADETS_E3
 python causal_screening_prototype.py orthrus CADETS_E3 --epoch 15

Requirements:
 - Training must have been run (edge_losses CSVs exist in artifacts)
 - No model loading or CUDA needed — runs on CPU in seconds
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.utils.utils import log

# =============================================================================
# Backdoor_06 malicious nodes (index_id from ground truth CSV)
# =============================================================================
BACKDOOR_06_MALICIOUS_IDS = {211776, 26277, 26281, 26254, 26318, 26357, 228725, 1219519}

# Attack time window for Backdoor_06 (from config.py attack_to_time_window)
ATTACK_TW_START_NS = 1523026800 * 10**9 # 2018-04-06 11:20:00 UTC approx
ATTACK_TW_END_NS = 1523030940 * 10**9 # 2018-04-06 12:09:00 UTC approx

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
 # Return the most recently modified one
 matches.sort(key=os.path.getmtime, reverse=True)
 return matches[0]

def list_available_epochs(edge_losses_dir):
 """List available model epochs in the edge_losses/test directory."""
 epochs = []
 for entry in os.listdir(edge_losses_dir):
 if entry.startswith("model_epoch_"):
 try:
 ep = int(entry.replace("model_epoch_", ""))
 epochs.append(ep)
 except ValueError:
 pass
 epochs.sort()
 return epochs

def load_edge_losses(edge_losses_dir, epoch=None):
 """Load all edge losses from CSV files for a given epoch.

 Args:
 edge_losses_dir: path to .../edge_losses/test/
 epoch: epoch number, or None to use the latest

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
 log(f"Loading {len(csv_files)} time-window CSV files from {epoch_dir}")

 # Fast load: concat all CSVs, convert to list of dicts in one shot
 dfs = [pd.read_csv(f) for f in csv_files]
 df = pd.concat(dfs, ignore_index=True)

 # Rename columns to match expected format
 df = df.rename(columns={"srcnode": "src", "dstnode": "dst", "time": "t"})
 all_edges = df.to_dict("records")

 log(f"Loaded {len(all_edges)} test edges")
 return all_edges, epoch

def filter_to_subgraph(all_edges, malicious_ids, hop=2):
 """Filter edges to the k-hop neighborhood of malicious nodes."""
 malicious_set = set(malicious_ids)

 # Build adjacency
 adjacency = defaultdict(set)
 for e in all_edges:
 adjacency[e["src"]].add(e["dst"])
 adjacency[e["dst"]].add(e["src"])

 # BFS for k-hop neighborhood
 current_frontier = malicious_set.copy()
 neighborhood = malicious_set.copy()
 for _ in range(hop):
 next_frontier = set()
 for node in current_frontier:
 next_frontier.update(adjacency.get(node, set()))
 next_frontier -= neighborhood
 neighborhood.update(next_frontier)
 current_frontier = next_frontier

 # Filter edges to neighborhood
 subgraph_edges = [
 e for e in all_edges
 if e["src"] in neighborhood or e["dst"] in neighborhood
 ]

 log(f"Subgraph: {len(neighborhood)} nodes, {len(subgraph_edges)} edges "
 f"(from {hop}-hop neighborhood of {len(malicious_set)} malicious nodes)")
 return subgraph_edges, neighborhood

def compute_aggregation_ice(subgraph_edges, malicious_ids, use_dst_loss=True):
 """Compute ICE using the exact aggregation-level formula.

 ICE(e, v) = max(0, score(v) - max(losses of v's OTHER edges))
 """
 malicious_set = set(malicious_ids)

 # Build node -> list of (loss, edge_key) mapping
 node_to_losses = defaultdict(list)
 for e in subgraph_edges:
 key = (e["src"], e["dst"], e["t"])
 node_to_losses[e["src"]].append((e["loss"], key))
 if use_dst_loss:
 node_to_losses[e["dst"]].append((e["loss"], key))

 # Baseline node scores
 node_scores = {}
 for node, loss_keys in node_to_losses.items():
 node_scores[node] = max(l for l, _ in loss_keys)

 # ICE per edge
 ice_results = {}
 for e in subgraph_edges:
 key = (e["src"], e["dst"], e["t"])
 ice_dict = {}
 for node in [e["src"], e["dst"]]:
 if node not in malicious_set:
 continue
 if use_dst_loss or node == e["src"]:
 other_losses = [l for l, k in node_to_losses[node] if k != key]
 new_score = max(other_losses) if other_losses else 0.0
 ice = node_scores[node] - new_score
 ice_dict[node] = ice
 if ice_dict:
 ice_results[key] = ice_dict

 return ice_results, node_scores

def define_attack_edges(subgraph_edges, malicious_ids, definition="B"):
 """Define which edges are 'attack edges' based on node-level ground truth."""
 malicious_set = set(malicious_ids)
 attack_edges = set()
 for e in subgraph_edges:
 key = (e["src"], e["dst"], e["t"])
 src_mal = e["src"] in malicious_set
 dst_mal = e["dst"] in malicious_set
 if definition == "A":
 is_attack = src_mal and dst_mal
 elif definition == "B":
 is_attack = src_mal or dst_mal
 elif definition == "C":
 in_tw = ATTACK_TW_START_NS <= e["t"] <= ATTACK_TW_END_NS
 is_attack = (src_mal or dst_mal) and in_tw
 else:
 raise ValueError(f"Unknown definition: {definition}")
 if is_attack:
 attack_edges.add(key)
 return attack_edges

def evaluate_ci_topk(ice_results, attack_edges, top_k=10):
 """Rank edges by CI and evaluate precision."""
 ci_scores = {}
 for edge_key, ice_dict in ice_results.items():
 ci = np.mean([abs(v) for v in ice_dict.values()])
 ci_scores[edge_key] = ci

 ranked = sorted(ci_scores.items(), key=lambda x: x[1], reverse=True)
 top_k_edges = ranked[:top_k]
 top_k_attack_count = sum(1 for edge_key, _ in top_k_edges if edge_key in attack_edges)
 precision = top_k_attack_count / min(top_k, len(top_k_edges)) if top_k_edges else 0.0

 precision_at = {}
 for k in [5, 10, 20, 50]:
 if len(ranked) >= k:
 topk = ranked[:k]
 hit = sum(1 for edge_key, _ in topk if edge_key in attack_edges)
 precision_at[k] = hit / k

 return {
 "top_k_edges": [
 {"edge": list(key), "ci": float(ci), "is_attack": key in attack_edges}
 for key, ci in top_k_edges
 ],
 "precision_at_10": precision,
 "precision_at": {k: float(v) for k, v in precision_at.items()},
 "total_subgraph_edges": len(ci_scores),
 "total_attack_edges": len(attack_edges),
 "attack_edge_ratio": len(attack_edges) / max(1, len(ci_scores)),
 }

def main():
 log("=" * 70)
 log("Causal Screening Prototype (CSV-based, no model loading needed)")
 log("=" * 70)

 # 1. Parse config
 args = get_runtime_required_args()
 cfg = get_yml_cfg(args)
 dataset_name = cfg.dataset.name
 log(f"Dataset: {dataset_name}")

 # 2. Find edge losses
 log("\n--- Finding edge losses ---")
 edge_losses_dir = find_edge_losses_dir(cfg)
 log(f"Edge losses dir: {edge_losses_dir}")

 epochs = list_available_epochs(edge_losses_dir)
 log(f"Available epochs: {epochs}")

 # Use --epoch if specified, else latest
 epoch = getattr(args, "training.num_epochs", None)
 # Try to get epoch from args if user passed --epoch
 specified_epoch = None
 for i, arg in enumerate(sys.argv):
 if arg == "--epoch" and i + 1 < len(sys.argv):
 specified_epoch = int(sys.argv[i + 1])
 break

 all_edges, used_epoch = load_edge_losses(edge_losses_dir, specified_epoch)

 if len(all_edges) == 0:
 log("ERROR: No edges loaded. Check that training produced edge_losses CSVs.")
 sys.exit(1)

 # 3. Ground truth
 malicious_ids = BACKDOOR_06_MALICIOUS_IDS
 test_nodes = set()
 for e in all_edges:
 test_nodes.add(e["src"])
 test_nodes.add(e["dst"])
 found_malicious = malicious_ids & test_nodes
 log(f"\nMalicious nodes in test set: {len(found_malicious)}/{len(malicious_ids)}")
 if len(found_malicious) == 0:
 log("WARNING: None of the Backdoor_06 malicious nodes found in test edges!")

 # 4. Subgraph extraction
 log("\n--- Extracting Backdoor_06 subgraph (2-hop neighborhood) ---")
 subgraph_edges, neighborhood = filter_to_subgraph(all_edges, malicious_ids, hop=2)

 if len(subgraph_edges) == 0:
 log("ERROR: Subgraph is empty.")
 sys.exit(1)

 # 5. Compute ICE
 log("\n--- Computing ICE scores ---")
 use_dst = True # matches orthrus.yml use_dst_node_loss: True
 ice_results, node_scores = compute_aggregation_ice(subgraph_edges, malicious_ids, use_dst_loss=use_dst)
 log(f"Computed ICE for {len(ice_results)} edges touching malicious nodes")

 log("\nBaseline node anomaly scores for malicious nodes:")
 for nid in sorted(malicious_ids):
 score = node_scores.get(nid, None)
 if score is not None:
 log(f" node {nid}: score={score:.4f}")
 else:
 log(f" node {nid}: NOT IN SUBGRAPH")

 # 6. Evaluate
 log("\n--- Evaluating CI top-k precision ---")

 for def_name in ["A", "B", "C"]:
 attack_edges = define_attack_edges(subgraph_edges, malicious_ids, definition=def_name)
 if len(attack_edges) == 0:
 log(f"\nDefinition {def_name}: 0 attack edges, skipping.")
 continue

 results = evaluate_ci_topk(ice_results, attack_edges, top_k=10)

 log(f"\n{'='*50}")
 log(f"Definition {def_name}: {len(attack_edges)} attack edges "
 f"({results['attack_edge_ratio']:.1%} of subgraph)")
 log(f"{'='*50}")
 log(f"\nCI Top-10 edges:")
 for i, item in enumerate(results["top_k_edges"]):
 tag = "ATTACK" if item["is_attack"] else "benign"
 log(f" #{i+1}: edge={item['edge']} CI={item['ci']:.4f} [{tag}]")

 log(f"\n*** Precision@10 = {results['precision_at_10']:.1%} ***")
 log(f"Precision@5={results['precision_at'].get(5, 'N/A')}, "
 f"@20={results['precision_at'].get(20, 'N/A')}, "
 f"@50={results['precision_at'].get(50, 'N/A')}")

 p10 = results["precision_at_10"]
 if p10 >= 0.7:
 log(f"\n>>> GO: Precision@10={p10:.1%} >= 0.7. Hypothesis CONFIRMED. <<<")
 elif p10 >= 0.5:
 log(f"\n>>> BORDERLINE: Precision@10={p10:.1%}. Investigate before deciding. <<<")
 else:
 log(f"\n>>> NO-GO: Precision@10={p10:.1%} < 0.5. Hypothesis may not hold. <<<")

 # Save results
 out_dir = cfg._artifact_dir
 os.makedirs(out_dir, exist_ok=True)
 out_file = os.path.join(out_dir, f"causal_screening_epoch{used_epoch}_def{def_name}.json")
 with open(out_file, "w") as f:
 json.dump(results, f, indent=2)
 log(f"Results saved to {out_file}")

 log("\n" + "=" * 70)
 log("Causal Screening prototype finished.")
 log("=" * 70)

if __name__ == "__main__":
 main()
