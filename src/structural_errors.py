#!/usr/bin/env python3
"""Summarize structural-group errors, similarity cutoffs, and paired-model uncertainty."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--predictions", type=Path, required=True, help="Folder of five cohort predictions"
)
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
PRED = ["pred_" + m + "_p" for m in MODS]
paths = {
    name: args.predictions / (name + ".csv.gz")
    for name in (
        "broad_fish",
        "broad_water_flea",
        "broad_algae",
        "fathead_minnow",
        "daphnia_magna",
    )
}
assert all(x.exists() for x in paths.values())
features = (
    "fragment_organophosphorus",
    "fragment_aliphatic_halogen",
    "fragment_aromatic_halogen",
    "fragment_phenol",
    "fragment_aliphatic_alcohol",
    "fragment_aromatic_amine",
    "fragment_amine_n_sp3_only",
    "fragment_aromatic_nitro",
    "fragment_carboxylic_acid",
    "fragment_carboxylic_ester",
)
cohort = []
model_rows = []
feature_rows = []
model_feature_rows = []
all_chem = set()
cutoff_rows = []
model_vote_rows = []
strata_rows = []
for name, path in paths.items():
    q = pd.read_csv(path)
    assert q.conn.is_unique and "OEJNXTAZZBRGDN" not in set(q.conn)
    all_chem.update(q.conn)
    d = q[q.ad_inside_5nn].copy()
    error = d.p_value.to_numpy()[:, None] - d[PRED].to_numpy()
    under = error > 1
    absolute = np.abs(error) > 1
    group = d.fragment_organophosphorus.to_numpy(bool)
    similarity = d.ad_mean_5nn_similarity.to_numpy(float)
    reference = d.p_value.to_numpy(float)
    predictions = d[PRED].to_numpy(float)
    q1, q3 = np.quantile(predictions, [0.25, 0.75], axis=1)
    narrow = q3 - q1 <= np.log10(2.0)
    median_error = np.abs(np.median(predictions, axis=1) - reference) > 1
    sim_bin = pd.qcut(
        d.ad_nearest_similarity, 4, labels=False, duplicates="drop"
    ).to_numpy()
    tox_bin = pd.qcut(d.p_value, 4, labels=False, duplicates="drop").to_numpy()
    strata = 10 * tox_bin + sim_bin
    for j, model in enumerate(MODS):
        weighted_differences = []
        weights = []
        for value in np.unique(strata[np.isfinite(strata)]):
            a = (strata == value) & group
            b = (strata == value) & ~group
            if a.any() and b.any():
                weighted_differences.append(
                    np.abs(error[a, j]).mean() - np.abs(error[b, j]).mean()
                )
                weights.append(a.sum())
        if weights:
            strata_rows.append(
                {
                    "dataset": name,
                    "model": model,
                    "unadjusted_mae_difference": np.abs(error[group, j]).mean()
                    - np.abs(error[~group, j]).mean(),
                    "stratified_mae_difference": np.average(
                        weighted_differences, weights=weights
                    ),
                }
            )
    for cutoff, supported in [
        ("training", np.ones(len(d), bool)),
        *[(c, similarity >= c) for c in (0.30, 0.40, 0.50)],
    ]:
        for group_name, mask in (
            ("all", np.ones(len(d), bool)),
            ("organophosphorus", group),
            ("other", ~group),
        ):
            keep = mask & supported
            if not keep.any():
                continue
            n_errors = absolute[keep].sum(axis=1)
            cutoff_rows.append(
                {
                    "dataset": name,
                    "cutoff": cutoff,
                    "group": group_name,
                    "retained": keep.sum(),
                    "median_large_error_rate": np.median(absolute[keep].mean(axis=0)),
                    "narrow_prediction_spread": (keep & narrow).sum(),
                    "narrow_spread_large_median_error": (
                        keep & narrow & median_error
                    ).sum(),
                }
            )
            if group_name != "all":
                model_vote_rows.append(
                    {
                        "dataset": name,
                        "cutoff": cutoff,
                        "group": group_name,
                        "zero_models": (n_errors == 0).sum(),
                        "one_to_four": ((n_errors >= 1) & (n_errors <= 4)).sum(),
                        "five_to_nine": ((n_errors >= 5) & (n_errors <= 9)).sum(),
                        "ten_models": (n_errors == 10).sum(),
                    }
                )
    cohort.append(
        {
            "dataset": name,
            "total": len(q),
            "inside_ad": len(d),
            "at_least_6_models_under_10fold": int((under.sum(axis=1) >= 6).sum()),
            "all_10_models_under_10fold": int((under.sum(axis=1) == 10).sum()),
            "at_least_6_models_absolute_10fold": int((absolute.sum(axis=1) >= 6).sum()),
        }
    )
    for j, model in enumerate(MODS):
        model_rows.append(
            {
                "dataset": name,
                "model": model,
                "n": len(d),
                "mae_log10": float(np.abs(error[:, j]).mean()),
                "rmse_log10": float(np.sqrt((error[:, j] ** 2).mean())),
                "under_10fold_rate": float(under[:, j].mean()),
                "absolute_10fold_rate": float(absolute[:, j].mean()),
            }
        )
    rng = np.random.default_rng(6192601)
    for feature in features:
        mask = d[feature].to_numpy(dtype=bool)
        if mask.sum() < 5 or (~mask).sum() < 5:
            continue
        rates_a = under[mask].mean(axis=0)
        rates_b = under[~mask].mean(axis=0)
        abs_a = absolute[mask].mean(axis=0)
        abs_b = absolute[~mask].mean(axis=0)
        a = under[mask]
        b = under[~mask]
        a_abs = absolute[mask]
        b_abs = absolute[~mask]
        diffs = []
        abs_diffs = []
        for _ in range(2000):
            ia = rng.integers(len(a), size=len(a))
            ib = rng.integers(len(b), size=len(b))
            aa = a[ia].mean(axis=0)
            bb = b[ib].mean(axis=0)
            diffs.append(float(np.median(aa - bb)))
            abs_diffs.append(
                float(np.median(a_abs[ia].mean(axis=0) - b_abs[ib].mean(axis=0)))
            )
        feature_rows.append(
            {
                "dataset": name,
                "fragment": feature,
                "group_n": int(mask.sum()),
                "other_n": int((~mask).sum()),
                "group_reference_p_median": float(d.loc[mask, "p_value"].median()),
                "other_reference_p_median": float(d.loc[~mask, "p_value"].median()),
                "group_similarity_median": float(
                    d.loc[mask, "ad_mean_5nn_similarity"].median()
                ),
                "other_similarity_median": float(
                    d.loc[~mask, "ad_mean_5nn_similarity"].median()
                ),
                "group_model_median_under_rate": float(np.median(rates_a)),
                "other_model_median_under_rate": float(np.median(rates_b)),
                "model_median_paired_rate_difference": float(
                    np.median(rates_a - rates_b)
                ),
                "conditional_chemical_bootstrap_95_low": float(
                    np.quantile(diffs, 0.025)
                ),
                "conditional_chemical_bootstrap_95_high": float(
                    np.quantile(diffs, 0.975)
                ),
                "group_model_median_absolute_error_rate": float(np.median(abs_a)),
                "other_model_median_absolute_error_rate": float(np.median(abs_b)),
                "model_median_paired_absolute_rate_difference": float(
                    np.median(abs_a - abs_b)
                ),
                "absolute_bootstrap_95_low": float(np.quantile(abs_diffs, 0.025)),
                "absolute_bootstrap_95_high": float(np.quantile(abs_diffs, 0.975)),
                "group_at_least_6_models_under_n": int((a.sum(axis=1) >= 6).sum()),
                "other_at_least_6_models_under_n": int((b.sum(axis=1) >= 6).sum()),
            }
        )
        for j, model in enumerate(MODS):
            model_feature_rows.append(
                {
                    "dataset": name,
                    "fragment": feature,
                    "model": model,
                    "group_n": int(mask.sum()),
                    "other_n": int((~mask).sum()),
                    "group_under_10fold_n": int(a[:, j].sum()),
                    "other_under_10fold_n": int(b[:, j].sum()),
                    "group_under_10fold_rate": float(rates_a[j]),
                    "other_under_10fold_rate": float(rates_b[j]),
                    "group_absolute_10fold_rate": float(abs_a[j]),
                    "other_absolute_10fold_rate": float(abs_b[j]),
                }
            )
pd.DataFrame(cutoff_rows).to_csv(output / "similarity_cutoffs.csv", index=False)
pd.DataFrame(model_vote_rows).to_csv(output / "model_error_counts.csv", index=False)
pd.DataFrame(strata_rows).to_csv(output / "stratified_mae.csv", index=False)
pd.DataFrame(cohort).to_csv(output / "15_cohort_error_summary.csv", index=False)
pd.DataFrame(model_rows).to_csv(output / "15_model_metrics.csv", index=False)
pd.DataFrame(feature_rows).to_csv(output / "15_fragment_error_summary.csv", index=False)
pd.DataFrame(model_feature_rows).to_csv(
    output / "15_fragment_by_model.csv", index=False
)
meta = {
    "distinct_connectivities_all_five_datasets": len(all_chem),
    "datasets": cohort,
    "errors": "p_reference minus p_prediction > 1 means >10-fold toxicity underprediction",
    "reference": "group percentages summarize 10 model-specific fractions; median difference pairs model-specific fractions.",
    "bootstrap": "2000 stratified chemical resamples with fixed fitted models and fixed datasets, seed 6192601",
    "limits": [
        "Fragments overlap, groups and endpoints were previously inspected, and comparisons are exploratory.",
        "Reference potency can differ across groups; raw differences do not establish causal effects of a fragment.",
        "Bootstrap intervals do not propagate measurement uncertainty or model retraining.",
    ],
}
(output / "15_structural_error_manifest.json").write_text(
    json.dumps(meta, indent=2) + "\n"
)
print(json.dumps(meta, indent=2), flush=True)
