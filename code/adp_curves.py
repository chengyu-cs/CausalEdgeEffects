"""Plot ADP curves for paper: Baseline vs Causal-Aware across 3 datasets."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg") # no display needed
import matplotlib.pyplot as plt

ARTIFACTS = os.environ.get("ARTIFACT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "artifacts"))
DATASETS = [
 ("adp_cadets_e3.json", "CADETS_E3", "Epoch 13"),
 ("adp_theia_e3.json", "THEIA_E3", "Epoch 19"),
 ("adp_clearscope_e3.json", "CLEARSCOPE_E3", "Epoch 19"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (filename, dataset_name, epoch_label) in zip(axes, DATASETS):
 filepath = os.path.join(ARTIFACTS, filename)
 if not os.path.isfile(filepath):
 print(f"WARNING: {filepath} not found, skipping {dataset_name}")
 ax.set_title(f"{dataset_name} (missing)")
 continue

 with open(filepath) as f:
 data = json.load(f)

 curves = data["task_a"]["adp_curves"]
 base = curves["baseline"]
 ca = curves["causal_aware"]

 # Plot curves
 ax.plot(base["precisions"], base["detected_pcts"],
 "r--", linewidth=2, label=f"Baseline (ADP={data['task_a']['baseline']['adp']})")
 ax.plot(ca["precisions"], ca["detected_pcts"],
 "b-", linewidth=2, label=f"Causal-Aware (ADP={data['task_a']['causal_aware']['adp']})")

 ax.set_xlabel("Precision", fontsize=12)
 ax.set_ylabel("% Attacks Detected", fontsize=12)
 ax.set_title(f"{dataset_name} ({epoch_label})", fontsize=13)
 ax.set_xlim(-0.05, 1.05)
 ax.set_ylim(-5, 105)
 ax.legend(fontsize=9, loc="lower right")
 ax.grid(True, alpha=0.3)

plt.tight_layout()
output_png = os.path.join(ARTIFACTS, "adp_curves.png")
output_svg = os.path.join(ARTIFACTS, "adp_curves.svg")
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.savefig(output_svg, bbox_inches="tight")
print(f"Saved: {output_png}")
print(f"Saved: {output_svg}")
