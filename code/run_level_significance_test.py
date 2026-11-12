#!/usr/bin/env python3
"""
Experiment C: Statistical Significance Test

Performs paired Wilcoxon signed-rank test on baseline vs CA ADP.
Reads from the no-GT extension summary CSV or any CSV with baseline_adp and ca_gt_adp columns.

Usage:
 python experiments/run_level_significance_test.py ../../data/tests/summary.csv

 # Or if you have the no-GT extension summary:
 python experiments/run_level_significance_test.py ../results/no_gt_extension/summary.csv

Output:
 Prints test statistics + saves to ../results/exp_c_significance.json
"""
import json
import sys
from pathlib import Path

import numpy as np

def main():
 if len(sys.argv) < 2:
 print("Usage: python experiments/run_level_significance_test.py <summary.csv>")
 print(" The CSV must have columns: baseline_adp, ca_gt_adp")
 sys.exit(1)

 csv_path = sys.argv[1]
 if not Path(csv_path).exists():
 print(f"ERROR: {csv_path} not found")
 sys.exit(1)

 import pandas as pd
 from scipy.stats import wilcoxon

 df = pd.read_csv(csv_path)
 print(f"Loaded {len(df)} rows from {csv_path}")
 print(f"Columns: {list(df.columns)}")

 # Try to find baseline and CA columns
 baseline_col = None
 ca_col = None
 for b in ["baseline_adp", "Baseline", "baseline"]:
 if b in df.columns:
 baseline_col = b
 break
 for c in ["ca_gt_adp", "ca_adp", "CA", "ca"]:
 if c in df.columns:
 ca_col = c
 break

 if not baseline_col or not ca_col:
 print(f"ERROR: Could not find baseline/CA columns. Available: {list(df.columns)}")
 sys.exit(1)

 baseline = df[baseline_col].dropna().values
 ca = df[ca_col].dropna().values
 n = min(len(baseline), len(ca))
 baseline = baseline[:n]
 ca = ca[:n]

 print(f"\n{'='*60}")
 print(f"Paired Wilcoxon Signed-Rank Test (n={n})")
 print(f"{'='*60}")
 print(f" Baseline ADP: mean={np.mean(baseline):.4f}, std={np.std(baseline):.4f}")
 print(f" CA ADP: mean={np.mean(ca):.4f}, std={np.std(ca):.4f}")
 print(f" Mean delta: {np.mean(ca - baseline):+.4f}")

 # Wilcoxon signed-rank test (one-sided: CA > Baseline)
 try:
 stat, p_value = wilcoxon(ca, baseline, alternative="greater")
 print(f"\n Wilcoxon stat = {stat:.1f}")
 print(f" p-value = {p_value:.2e}")
 print(f" Significant (p<0.001)? {'YES' if p_value < 0.001 else 'NO'}")
 except Exception as e:
 stat, p_value = None, None
 print(f" Wilcoxon test failed: {e}")

 # Effect size: Cohen's d for paired samples
 deltas = ca - baseline
 std_d = np.std(deltas, ddof=1) if len(deltas) > 1 else 1.0
 cohens_d = np.mean(deltas) / std_d if std_d > 0 else float("inf")
 print(f"\n Cohen's d = {cohens_d:.3f}")
 if abs(cohens_d) > 0.8:
 print(f" Effect size: LARGE (>0.8)")
 elif abs(cohens_d) > 0.5:
 print(f" Effect size: MEDIUM (>0.5)")
 else:
 print(f" Effect size: SMALL")

 # Cliff's delta (non-parametric effect size)
 n_pairs = len(deltas)
 pos = sum(1 for d in deltas if d > 0)
 neg = sum(1 for d in deltas if d < 0)
 cliffs_d = (pos - neg) / n_pairs if n_pairs > 0 else 0
 print(f"\n Cliff's delta = {cliffs_d:.3f}")
 print(f" Positive pairs: {pos}/{n_pairs} ({pos/n_pairs*100:.1f}%)")

 # Summary for paper
 result = {
 "n_checkpoints": n,
 "baseline_mean": round(float(np.mean(baseline)), 4),
 "baseline_std": round(float(np.std(baseline)), 4),
 "ca_mean": round(float(np.mean(ca)), 4),
 "ca_std": round(float(np.std(ca)), 4),
 "mean_delta": round(float(np.mean(deltas)), 4),
 "wilcoxon_stat": round(float(stat), 1) if stat else None,
 "p_value": float(p_value) if p_value else None,
 "cohens_d": round(float(cohens_d), 3),
 "cliffs_delta": round(float(cliffs_d), 3),
 "n_improved": pos,
 "improvement_rate": f"{pos}/{n} ({pos/n*100:.1f}%)",
 }

 out_path = Path(__file__).parent.parent.parent / "results" / "exp_c_significance.json"
 out_path.parent.mkdir(exist_ok=True)
 out_path.write_text(json.dumps(result, indent=2))
 print(f"\nResults saved: {out_path}")

 # Print paper-ready sentence
 print(f"\n{'='*60}")
 print("Paper-ready sentence:")
 print(f"{'='*60}")
 if p_value and p_value < 0.001:
 print(f" A paired Wilcoxon signed-rank test confirms that CA significantly "
 f"improves ADP over the baseline (p < 0.001, Cohen's d = {cohens_d:.2f}, "
 f"n = {n}).")
 elif p_value:
 print(f" A paired Wilcoxon signed-rank test shows CA improves ADP "
 f"(p = {p_value:.4f}, Cohen's d = {cohens_d:.2f}, n = {n}).")

if __name__ == "__main__":
 main()
