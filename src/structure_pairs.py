#!/usr/bin/env python3
"""Recalculate single-species matched error contrasts and same-fold pairs."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from rdkit import DataStructs
from molecular_models import get_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
output = args.output
output.mkdir(parents=True, exist_ok=True)
MODS = (
    "random_forest",
    "lightgbm",
    "xgboost",
    "svr",
    "extra_trees",
    "ridge",
    "catboost",
    "mlp",
    "gine",
    "gat",
)
d = pd.read_csv(args.predictions)
d = d[d.ad_inside_5nn].reset_index(drop=True)
fp, _, _ = get_data(pd.DataFrame({"smiles": d.smiles}))
PRED = ["pred_" + m + "_p" for m in MODS]
pred = d[PRED].to_numpy(float)
reference = d.p_value.to_numpy(float)
rows = []
for i in range(len(d)):
    sims = DataStructs.BulkTanimotoSimilarity(fp[i], fp[i + 1 :])
    for off, similarity in enumerate(sims, 1):
        j = i + off
        if (
            d.fold[i] != d.fold[j]
            or similarity < 0.5
            or abs(reference[i] - reference[j]) <= 1
        ):
            continue
        delta = reference[j] - reference[i]
        reverse = (pred[j] - pred[i]) * delta < 0
        rows.append(
            {
                "conn_a": d.conn[i],
                "conn_b": d.conn[j],
                "fold": int(d.fold[i]),
                "similarity": similarity,
                "reference_delta_b_minus_a": delta,
                "n_models_reversed": int(reverse.sum()),
                **{
                    "delta_" + m: float(pred[j, k] - pred[i, k])
                    for k, m in enumerate(MODS)
                },
            }
        )
pd.DataFrame(rows).to_csv(output / "25_same_fold_activity_pairs.csv", index=False)
pair = d.set_index("conn").loc[["MCWXGJITAZMZEV", "PZXOQEXFMJCDPG"]]
assert pair.fold.nunique() == 1
delta = pair.iloc[1][PRED].to_numpy(float) - pair.iloc[0][PRED].to_numpy(float)
pair_out = [
    {
        "model": "Measured",
        "dimethoate": float(pair.iloc[0].p_value),
        "omethoate": float(pair.iloc[1].p_value),
        "delta_pEC50_omethoate_minus_dimethoate": float(
            pair.iloc[1].p_value - pair.iloc[0].p_value
        ),
    }
]
pair_out += [
    {
        "model": m,
        "dimethoate": float(pair.iloc[0]["pred_" + m + "_p"]),
        "omethoate": float(pair.iloc[1]["pred_" + m + "_p"]),
        "delta_pEC50_omethoate_minus_dimethoate": float(delta[j]),
    }
    for j, m in enumerate(MODS)
]
pd.DataFrame(pair_out).to_csv(output / "25_dimethoate_pair.csv", index=False)

# Same quartile-by-quartile weighting as the original analysis: OP counts are
# the weights in strata where OP and other chemicals are both represented.
flag = d.fragment_organophosphorus.to_numpy(bool)
sim = pd.qcut(d.ad_nearest_similarity, 4, labels=False, duplicates="drop").to_numpy()
records = []
for model in MODS:
    for kind, potency in [
        ("observed", d.p_value),
        ("predicted", d["pred_" + model + "_p"]),
    ]:
        pb = pd.qcut(potency, 4, labels=False, duplicates="drop").to_numpy()
        ids = 10 * pb + sim
        groups = []
        for s in np.unique(ids[np.isfinite(ids)]):
            index = np.flatnonzero(ids == s)
            a = index[flag[index]]
            b = index[~flag[index]]
            if len(a) and len(b):
                groups.append((a, b))
        supported = sum(len(a) for a, b in groups)
        err = np.abs(d.p_value.to_numpy() - d["pred_" + model + "_p"].to_numpy())
        adjusted = float(
            sum(len(a) * (err[a].mean() - err[b].mean()) for a, b in groups) / supported
        )
        records.append(
            {
                "endpoint": "daphnia",
                "model": model,
                "feature": "fragment_organophosphorus",
                "metric": "absolute_log_error",
                "stratified_by": kind,
                "feature_n": int(flag.sum()),
                "other_n": int((~flag).sum()),
                "feature_supported_n": supported,
                "supported_strata": len(groups),
                "unadjusted_feature_mean": float(err[flag].mean()),
                "unadjusted_other_mean": float(err[~flag].mean()),
                "adjusted_difference": adjusted,
            }
        )
z = pd.DataFrame(records)
z.to_csv(output / "25_single_matched_contrasts.csv.gz", index=False, compression="gzip")
observed = z[z.stratified_by == "observed"]
stricter = d[d.ad_mean_5nn_similarity.ge(0.4)]
op = stricter[stricter.fragment_organophosphorus]
op_errors = np.abs(op.p_value.to_numpy()[:, None] - op[PRED].to_numpy()) > 1
summary = {
    "domain_chemicals": len(d),
    "qualified_same_fold_pairs": len(rows),
    "pairs_reversed_by_all_ten_models": sum(r["n_models_reversed"] == 10 for r in rows),
    "dimethoate_omethoate_models_reversed": int((delta < 0).sum()),
    "dimethoate_omethoate_reference_difference": pair_out[0][
        "delta_pEC50_omethoate_minus_dimethoate"
    ],
    "observed_stratification_unadjusted_median_mae_difference": float(
        np.median(observed.unadjusted_feature_mean - observed.unadjusted_other_mean)
    ),
    "observed_stratification_adjusted_median_mae_difference": float(
        observed.adjusted_difference.median()
    ),
    "similarity_0_4_op_chemicals": len(op),
    "similarity_0_4_op_at_least_5_models_with_large_error": int(
        (op_errors.sum(axis=1) >= 5).sum()
    ),
    "limits": [
        "Pairs and thresholds were selected after inspection.",
        "Measured-toxicity stratification is explanatory and not deployable before a toxicity value is known.",
    ],
}
(output / "25_pair_strata_manifest.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2), flush=True)
