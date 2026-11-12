#!/usr/bin/env python3
"""E2a: independent verification of /C2 (nth(1) indexing bug) and
corrected side-aware closed-form recomputation across all three architectures,
plus P1-stage3 audit toy and plateau structure stats. Produces the v2 summary
with independent-edge dedup and coverage/same-sign/magnitude reporting
(/M3 fixes).
"""
import json
import os

import numpy as np
import pandas as pd

WS = os.environ.get("CAUSAL_WORKSPACE", ".")
OUT = os.path.join(WS, "results/closed_form_validation")
R1 = os.path.join(WS, "results/e1a")

def second_max(series):
 s = series.nlargest(2)
 return s.iloc[-1] if len(s) >= 2 else np.nan

def corrected_closed(base):
 """Side-aware closed-form with (a) correct second-max indexing by node id,
 (b) effect referenced to the endpoint's pre-deletion node max."""
 src_max = base.groupby("srcnode")["loss"].max()
 dst_max = base.groupby("dstnode")["loss"].max()
 node_max = pd.concat([src_max, dst_max]).groupby(level=0).max()
 src_second = base.groupby("srcnode")["loss"].apply(second_max)
 dst_second = base.groupby("dstnode")["loss"].apply(second_max)

 df = base.copy()
 # new node max if e (row) is removed, per side; NaN side = side loses its
 # only/max candidate
 new_u = pd.concat([src_second, dst_max], axis=1).max(axis=1) # indexed by node
 new_v = pd.concat([dst_second, src_max], axis=1).max(axis=1)
 ls = df["loss"].to_numpy()
 is_u_max = ls == df["srcnode"].map(src_max).to_numpy()
 is_v_max = ls == df["dstnode"].map(dst_max).to_numpy()
 # effect on node max (<=0), NaN => node loses ALL edges -> full withdrawal
 raw_u = df["srcnode"].map(new_u).to_numpy() - df["srcnode"].map(node_max).to_numpy()
 raw_v = df["dstnode"].map(new_v).to_numpy() - df["dstnode"].map(node_max).to_numpy()
 eff_u = np.where(is_u_max, raw_u, 0.0)
 eff_v = np.where(is_v_max, raw_v, 0.0)
 # node exits (no surviving edge): full withdrawal to 0
 exits_u = is_u_max & df["srcnode"].map(new_u).isna().to_numpy()
 exits_v = is_v_max & df["dstnode"].map(new_v).isna().to_numpy()
 eff_u = np.where(exits_u, -df["srcnode"].map(node_max).to_numpy(), eff_u)
 eff_v = np.where(exits_v, -df["dstnode"].map(node_max).to_numpy(), eff_v)
 df["eff_u_corr"] = eff_u
 df["eff_v_corr"] = eff_v
 df["ci_corr"] = np.where((eff_u <= eff_v) & (eff_u < 0), -eff_u,
 np.where(eff_v < 0, -eff_v, np.nan))
 return df, node_max

def closed_endpoint_of(key, base, corr):
 """Recompute corrected closed endpoint delta for one archived edge,
 matched by (src,dst,time,edge_type); key is an int dict (no float upcast)."""
 m = base[(base.srcnode == key["src"]) & (base.dstnode == key["dst"])
 & (base.time == key["time"]) & (base.edge_type == key["edge_type"])]
 if len(m) == 0:
 return np.nan, np.nan
 r = corr.loc[m.index[0]]
 d = {}
 if r["eff_u_corr"] != 0.0:
 d[key["src"]] = r["eff_u_corr"]
 if r["eff_v_corr"] != 0.0:
 d[key["dst"]] = r["eff_v_corr"]
 return (min(d.values()) if d else 0.0), float(m.iloc[0]["loss"])

def main():
 os.makedirs(OUT, exist_ok=True)
 # ---------------- toy verification of the nth(1) bug () -----------
 toy = pd.DataFrame({"node": [10, 10, 10, 20, 20], "loss": [9, 5, 1, 3, 2]})
 s_bug = toy.sort_values(["node", "loss"], ascending=[True, False]).groupby("node")["loss"].nth(1)
 toy_report = {
 "env_pandas": pd.__version__,
 "nth1_index_labels": list(s_bug.index.astype(int)),
 "nth1_values": list(s_bug.values.astype(float)),
 "expected_if_groupkey_indexed": {"10": 5.0, "20": 2.0},
 "verdict": "nth(1) indexes by ORIGINAL ROW LABEL, not group key -> concat "
 "with node-id-indexed max series misaligns -> second becomes NaN "
 "-> fillna(0) -> closed-form degenerates to full withdrawal (max-0)",
 }
 # ---------------- P1 stage3 audit toy () ---------------------------
 long_df = pd.DataFrame({"node": [10, 10, 10, 20, 20], "loss": [9, 5, 1, 3, 2]})
 node_max_t = long_df.groupby("node")["loss"].max()
 long_sorted = long_df.sort_values(["node", "loss"], ascending=[True, False])
 top2 = long_sorted.groupby("node").head(2)
 node_second = top2.groupby("node")["loss"].nth(1)
 node_second = node_second.reindex(node_max_t.index).fillna(0.0)
 toy_report["stage3_node_second"] = {str(k): float(v) for k, v in node_second.items()}
 toy_report["stage3_expected_second"] = {"10": 5.0, "20": 2.0}
 toy_report["stage3_verdict"] = ("P1 stage3_causal_pipeline.py:258-261 node_second has "
 "the same misalignment: executed ICE ~ max-0 "
 "(full withdrawal) instead of documented max-second margin")
 with open(os.path.join(OUT, "e2a_toy_bug_verification.json"), "w") as f:
 json.dump(toy_report, f, indent=1)
 print(json.dumps(toy_report, indent=1), flush=True)

 # ---------------- corrected closed-form recomputation --------------------
 models = [
 ("kairos", "e1a_kairos_THEIA_E3_ep19"),
 ("orthrus", "e1a_orthrus_CADETS_E3_ep19"),
 ("velox", "e1a_velox_CADETS_E3_ep19"),
 ]
 all_rows = []
 for model, tag in models:
 base = pd.read_csv(os.path.join(R1, f"{tag}_baserep.csv"))
 arch = pd.read_csv(os.path.join(R1, f"{tag}_interventions.csv"))
 # merge the recovered 4-tuples (kairos only has a manifest; recompute for
 # all via deterministic selection reproduction)
 corr, node_max = corrected_closed(base)
 print(f"[{model}] baserep={len(base)} interventions={len(arch)}", flush=True)
 rows = []
 for _, r in arch.iterrows():
 key = None
 if model == "kairos":
 man = pd.read_csv(os.path.join(WS, "results/e2f/e2f_edge_manifest.csv"))
 mm = man[(man.stratum == r.stratum) & (man.k == r.k)]
 key = ({"src": int(mm["src"].iloc[0]), "dst": int(mm["dst"].iloc[0]),
 "time": int(mm["time"].iloc[0]),
 "edge_type": int(mm["edge_type"].iloc[0])}
 if len(mm) else None) # column access keeps int64 exact
 cc, loss_rec = (np.nan, np.nan) if key is None else closed_endpoint_of(key, base, corr)
 if key is None: # velox/orthrus: match by (src,dst,loss)
 m = base[(base.srcnode == r.src) & (base.dstnode == r.dst)
 & (np.abs(base.loss - r.loss) < 1e-9)]
 if len(m):
 rr = corr.loc[m.index[0]]
 d = {}
 if rr.eff_u_corr != 0.0:
 d[int(r.src)] = rr.eff_u_corr
 if rr.eff_v_corr != 0.0:
 d[int(r.dst)] = rr.eff_v_corr
 cc = min(d.values()) if d else 0.0
 rows.append({
 "model": model, "stratum": r.stratum, "k": int(r.k),
 "src": int(r.src), "dst": int(r.dst),
 "loss": float(r.loss),
 "closed_bugged_archived": float(r.closed_delta_endpoint),
 "true_archived": float(r.true_delta_endpoint),
 "closed_corrected": float(cc) if np.isfinite(cc) else np.nan,
 "abs_corr_minus_true": abs(cc - r.true_delta_endpoint) if np.isfinite(cc) else np.nan,
 "n_affected": int(r.n_affected),
 "unaff_keys_touched": int(r.unaff_keys_touched),
 "unaff_pct_of_test": float(r.unaff_keys_touched) / len(base),
 })
 d = pd.DataFrame(rows)
 d.to_csv(os.path.join(OUT, f"e2a_corrected_closed_{model}.csv"), index=False)
 all_rows.append(d)
 exact = (d["abs_corr_minus_true"] <= 1e-9).sum()
 print(f"[{model}] corrected==true (<=1e-9): {exact}/{len(d)} "
 f"max|diff|={d['abs_corr_minus_true'].max():.3e}", flush=True)
 del base, corr, node_max

 comb = pd.concat(all_rows, ignore_index=True)
 comb.to_csv(os.path.join(OUT, "e2a_corrected_closed_all.csv"), index=False)

 # plateau structure on kairos baserep (memory-compensation counter-evidence)
 base = pd.read_csv(os.path.join(R1, "e1a_kairos_THEIA_E3_ep19_baserep.csv"))
 u0 = 1194901
 g = base[(base.srcnode == u0) | (base.dstnode == u0)]["loss"]
 plateau = {
 "edges_in_12.60_12.90": int(((base.loss >= 12.60) & (base.loss <= 12.90)).sum()),
 "edges_ge_12.0": int((base.loss >= 12.0).sum()),
 "u_1194901_second_max": float(g.nlargest(2).iloc[-1]),
 "u_1194901_deg": int(len(g)),
 "note": "post-deletion endpoint max (archived true + second-max match) equals "
 "the PRE-EXISTING second max: plateau structure, not dynamic memory compensation",
 }
 with open(os.path.join(OUT, "e2a_plateau_structure.json"), "w") as f:
 json.dump(plateau, f, indent=1)
 print(json.dumps(plateau, indent=1), flush=True)

 # ---------------- v2 summary: dedup + coverage/sign/magnitude ------------
 dedup = comb.drop_duplicates(subset=["model", "src", "dst"]) if "time" not in comb else comb
 rows2 = []
 for (model, strat), d in comb.groupby(["model", "stratum"]):
 n = len(d)
 dd = d.drop_duplicates(subset=["src", "dst", "loss"])
 pred_nonzero = d[d.closed_corrected != 0]
 cov = len(pred_nonzero) / n
 same_sign = int((np.sign(pred_nonzero.closed_corrected)
 == np.sign(pred_nonzero.true_archived)).sum()) if len(pred_nonzero) else 0
 mag = (pred_nonzero.closed_corrected / pred_nonzero.true_archived.replace(0, np.nan)).abs()
 rows2.append({
 "model": model, "stratum": strat, "n_interventions": n,
 "n_independent_edges": len(dd),
 "pred_coverage_n_over_N": f"{len(pred_nonzero)}/{n}",
 "pred_coverage": round(cov, 3),
 "same_sign_n": same_sign,
 "abs_closed_over_true_median": float(mag.median()) if len(mag) else np.nan,
 "abs_closed_over_true_max": float(mag.max()) if len(mag) else np.nan,
 "true_delta_endpoint_median": float(d.true_archived.median()),
 "unaff_keys_pct_min": float(d.unaff_pct_of_test.min()),
 "unaff_keys_pct_max": float(d.unaff_pct_of_test.max()),
 })
 s2 = pd.DataFrame(rows2)
 s2.to_csv(os.path.join(OUT, "e2a_summary_v2.csv"), index=False)
 print(s2.to_string(), flush=True)
 print("E2A DONE", flush=True)

if __name__ == "__main__":
 main()
