#!/usr/bin/env python3
"""Replay published-coverage and independent water-flea error comparisons."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd

KEY = "SMILES_Canonical_RDKit"

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


def summarize(z, bootstrap=False):
    flag = z.OP_feature.to_numpy(bool)
    err = z[["error_" + m + "_p" for m in MODS]].to_numpy()
    n1 = int(flag.sum())
    n0 = int((~flag).sum())
    if not n1 or not n0:
        return {"OP_n": n1, "other_n": n0}
    out = {
        "OP_n": n1,
        "other_n": n0,
        "OP_reference_p_median": float(np.median(z.loc[flag, "reference_p_mmol_L"])),
        "other_reference_p_median": float(
            np.median(z.loc[~flag, "reference_p_mmol_L"])
        ),
        "metrics": {},
    }
    for label, v in [
        ("absolute_error_gt10", np.abs(err) > 1),
        ("underestimated_gt10", err < -1),
        ("overestimated_gt10", err > 1),
    ]:
        a = v[flag]
        b = v[~flag]
        ra = a.mean(axis=0)
        rb = b.mean(axis=0)
        entry = {
            "OP_median_percent": float(100 * np.median(ra)),
            "other_median_percent": float(100 * np.median(rb)),
            "paired_median_difference_pp": float(100 * np.median(ra - rb)),
            "model_counts": {
                m: {"OP": int(a[:, j].sum()), "other": int(b[:, j].sum())}
                for j, m in enumerate(MODS)
            },
        }
        if bootstrap:
            rng = np.random.default_rng(20261110)
            ia = rng.integers(0, len(a), size=(5000, len(a)))
            ib = rng.integers(0, len(b), size=(5000, len(b)))
            delta = np.median(100 * (a[ia].mean(axis=1) - b[ib].mean(axis=1)), axis=1)
            entry["conditional_chemical_bootstrap_95_pp"] = [
                float(x) for x in np.quantile(delta, [0.025, 0.975])
            ]
        out["metrics"][label] = entry
    return out


def matched(data, potency):
    flag = data.OP_feature.to_numpy(bool)
    sim = pd.qcut(data.nearest, 4, labels=False, duplicates="drop").to_numpy()
    result = []
    for m in MODS:
        val = (
            data.reference_p_mmol_L
            if potency == "measured"
            else data["pred_" + m + "_p"]
        )
        pb = pd.qcut(val, 4, labels=False, duplicates="drop").to_numpy()
        ids = 10 * pb + sim
        groups = []
        for x in np.unique(ids[np.isfinite(ids)]):
            ix = np.flatnonzero(ids == x)
            a = ix[flag[ix]]
            b = ix[~flag[ix]]
            if len(a) and len(b):
                groups.append((a, b))
        support = sum(len(a) for a, b in groups)
        if not support:
            raise ValueError("No supported strata for " + m)
        e = data["error_" + m + "_p"].to_numpy()
        vals = {
            name: float(
                sum(len(a) * (v[a].mean() - v[b].mean()) for a, b in groups) / support
            )
            for name, v in [
                ("under", e < -1),
                ("absolute", np.abs(e) > 1),
                ("absolute_error", np.abs(e)),
            ]
        }
        result.append(
            {
                "model": m,
                "supported_OP": support,
                "supported_strata": len(groups),
                **vals,
            }
        )
    return {
        "per_model": result,
        "under_adjusted_median_pp": 100 * statistics.median(x["under"] for x in result),
        "absolute_error_adjusted_median_p": statistics.median(
            x["absolute_error"] for x in result
        ),
    }


def stratified(chemical, columns):
    d = chemical.copy()
    for name in columns:
        d[name + "_quartile"] = pd.qcut(d[name], 4, labels=False, duplicates="drop")
    qcols = [name + "_quartile" for name in columns]
    grouped = d.groupby(qcols, dropna=False)
    rng = np.random.default_rng(20261106)
    repetitions = np.zeros(2000)
    total = 0
    estimate = 0.0
    retained = 0
    for _, group in grouped:
        a = group.loc[group.organophosphorus, "absolute_gt10"].to_numpy(float)
        b = group.loc[~group.organophosphorus, "absolute_gt10"].to_numpy(float)
        if not len(a) or not len(b):
            continue
        estimate += len(a) * (a.mean() - b.mean())
        repetitions += len(a) * (
            a[rng.integers(0, len(a), (2000, len(a)))].mean(axis=1)
            - b[rng.integers(0, len(b), (2000, len(b)))].mean(axis=1)
        )
        total += len(a)
        retained += 1
    if not total:
        return {
            "supported_organophosphorus": 0,
            "supported_strata": 0,
            "difference": None,
            "interval": None,
        }
    repetitions /= total
    return {
        "supported_organophosphorus": total,
        "supported_strata": retained,
        "difference": float(estimate / total),
        "interval": [
            float(np.quantile(repetitions, 0.025)),
            float(np.quantile(repetitions, 0.975)),
        ],
    }


def external_water_flea(d):
    inside = d[d.inside_AD].reset_index(drop=True)
    out = {
        "training_n": 401,
        "query_n": len(d),
        "inside_AD_n": len(inside),
        "conditions": {
            "all": summarize(d),
            "inside_AD": summarize(inside, True),
            "inside_AD_and_mean5_gte_0_4": summarize(inside[inside.mean5 >= 0.4]),
            "inside_AD_absent_all_other_project_cohorts": summarize(
                inside[inside.absent_other_project_cohorts]
            ),
        },
        "inside_AD_matched": {
            "measured_toxicity_and_similarity": matched(inside, "measured"),
            "predicted_toxicity_and_similarity": matched(inside, "predicted"),
        },
        "limits": [
            "The dataset is retrospective and source experiments may overlap the training literature.",
            "Grouping, data and analyses were inspected before this refit.",
            "Stratification on measured toxicity is explanatory, not available for an untested chemical.",
            "Chemical resampling holds the fitted models fixed.",
        ],
    }
    return out


def published_transformer(joined):
    # Each seed may contain multiple stereochemical SMILES with the same first
    # InChIKey block. Require all members and all repetitions to meet coverage
    # and exclude connectivities represented in any corresponding training set.
    selected = joined[joined.single_component_organic].copy()
    per_seed = selected.groupby(["conn", "seed"], as_index=False).agg(
        measured_log_mg_l=("measured_log_mg_l", "median"),
        predicted_log_mg_l=("predicted_log_mg_l", "median"),
        mw=("mw", "median"),
        nearest=("nearest", "mean"),
        organophosphorus=("organophosphorus", "max"),
        project_overlap=("project_overlap", "max"),
        in_coverage=("in_coverage", "min"),
        conn_in_training=("conn_in_training", "max"),
        n_smiles=(KEY, "size"),
    )
    chemical = per_seed.groupby("conn", as_index=False).agg(
        n_seeds=("seed", "nunique"),
        measured_log_mg_l=("measured_log_mg_l", "median"),
        predicted_log_mg_l=("predicted_log_mg_l", "mean"),
        mw=("mw", "median"),
        nearest=("nearest", "mean"),
        organophosphorus=("organophosphorus", "max"),
        project_overlap=("project_overlap", "max"),
        coverage_seeds=("in_coverage", "sum"),
        conn_in_training_any=("conn_in_training", "max"),
    )
    if chemical.n_seeds.ne(10).any():
        raise ValueError("Each chemical must have predictions from all 10 repetitions")
    chemical["toxicity_p_mmol_l"] = -chemical.measured_log_mg_l + np.log10(chemical.mw)
    chemical["error_log10"] = chemical.predicted_log_mg_l - chemical.measured_log_mg_l
    chemical["under_gt10"] = chemical.error_log10.gt(1)
    chemical["over_gt10"] = chemical.error_log10.lt(-1)
    chemical["absolute_gt10"] = chemical.error_log10.abs().gt(1)
    results = []
    for overlap in ["all", "project_nonoverlap"]:
        for domain in ["all", "at_least_9_of_10", "all_10"]:
            q = chemical[~chemical.conn_in_training_any].copy()
            if overlap == "project_nonoverlap":
                q = q[~q.project_overlap]
            if domain == "at_least_9_of_10":
                q = q[q.coverage_seeds.ge(9)]
            elif domain == "all_10":
                q = q[q.coverage_seeds.eq(10)]
            record = {
                "population": overlap,
                "coverage": domain,
                "n": len(q),
                "groups": {},
            }
            for is_present in (True, False):
                x = q[q.organophosphorus.eq(is_present)]
                label = "organophosphorus" if is_present else "other"
                record["groups"][label] = {
                    "n": len(x),
                    "absolute_gt10": int(x.absolute_gt10.sum()),
                    "under_gt10": int(x.under_gt10.sum()),
                    "over_gt10": int(x.over_gt10.sum()),
                    "mae": float(x.error_log10.abs().mean()) if len(x) else None,
                }
            record["toxicity_stratified"] = stratified(q, ["toxicity_p_mmol_l"])
            record["toxicity_similarity_stratified"] = stratified(
                q, ["toxicity_p_mmol_l", "nearest"]
            )
            eligible = set(q.conn)
            single = per_seed[per_seed.conn.isin(eligible)].copy()
            single["absolute_gt10"] = (
                single.predicted_log_mg_l - single.measured_log_mg_l
            ).abs() > 1
            ranges = {}
            for flag, name in [(True, "organophosphorus"), (False, "other")]:
                rates = (
                    single[single.organophosphorus.eq(flag)]
                    .groupby("seed")
                    .absolute_gt10.mean()
                )
                ranges[name] = {
                    "min": float(rates.min()),
                    "median": float(rates.median()),
                    "max": float(rates.max()),
                }
            record["single_repetition_rates"] = ranges
            results.append(record)
    report = {
        "status": "PASS",
        "repetitions": 10,
        "prediction_aggregation": "mean of the 10 out-of-fold per-seed predictions",
        "reference": "median distinct D. magna 48 h ITX log-concentrations",
        "domain": "same 5-neighbor training-only cutoff; restricted to RDKit-valid training structures",
        "query_selection": "single connected structure containing carbon",
        "connectivity_leak_excluded": True,
        "model_scope": "published pooled-invertebrate TRIDENT; not a single-species training model",
        "error_sign": "predicted log concentration minus observed log concentration",
        "total_connectivities": len(chemical),
        "any_connectivity_in_training": int(chemical.conn_in_training_any.sum()),
        "results": results,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", choices=["water-flea", "trident"])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    report = (
        external_water_flea(frame)
        if args.comparison == "water-flea"
        else published_transformer(frame)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
