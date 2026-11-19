#!/usr/bin/env python3
"""Replay national detection screening and threshold sensitivity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from molecular_models import get_data, top5

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--historical", type=Path, required=True)
parser.add_argument("--recent", type=Path, required=True)
parser.add_argument("--recent-with-estimates", type=Path)
parser.add_argument(
    "--train", type=Path, required=True, help="Curated training structures and p_value"
)
parser.add_argument(
    "--query",
    type=Path,
    required=True,
    help="Joint predictions and measured reference p_value",
)
parser.add_argument(
    "--selections",
    type=Path,
    required=True,
    help="Mapping from rule name to selected connectivities",
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
selection_lists = json.loads(args.selections.read_text())
train = pd.read_csv(args.train)
query = pd.read_csv(args.query).set_index("conn")
assert (
    train.conn.is_unique
    and query.index.is_unique
    and not set(train.conn) & set(query.index)
)
fp, _, _ = get_data(pd.DataFrame({"smiles": [*train.smiles, *query.smiles]}))
_, _, nn = top5(fp, np.arange(len(train)), np.arange(len(train), len(fp)))
guard = pd.Series(np.max(train.p_value.to_numpy()[nn], axis=1), index=query.index)
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
model_cols = [f"pred_{model}_p" for model in MODS]
rows = []
periods = [("historical", args.historical), ("recent_actual", args.recent)]
if args.recent_with_estimates:
    periods.append(("recent_including_estimates", args.recent_with_estimates))
for period, source in periods:
    detections = pd.read_csv(source, low_memory=False)
    assert detections.conn.isin(query.index).all()
    scale = (
        detections.concentration_ug_l.to_numpy(float)
        * 0.001
        / detections.mw.to_numpy(float)
    )
    reference = scale * 10 ** query.loc[detections.conn, "p_value"].to_numpy(float)
    predicted = scale[:, None] * 10 ** query.loc[detections.conn, model_cols].to_numpy(
        float
    )
    guard_quotient = scale * 10 ** guard.loc[detections.conn].to_numpy(float)
    alternative = {
        "model": predicted,
        "safety_factor_3": 3 * predicted,
        "safety_factor_10": 10 * predicted,
        "neighbor_guard": np.maximum(predicted, guard_quotient[:, None]),
    }
    for name, selected in selection_lists.items():
        tested = detections.conn.isin(selected).to_numpy()
        alternative[name] = np.where(tested[:, None], reference[:, None], predicted)
    for fraction in (0.5, 0.1, 0.01, 0.001):
        positive = (reference >= fraction) | np.isclose(
            reference, fraction, rtol=1e-12, atol=0
        )
        for name, quotients in alternative.items():
            flags = (
                (quotients >= fraction)
                | np.isclose(quotients, fraction, rtol=1e-12, atol=0)
            ).sum(axis=1)
            missed = positive & (flags < 5)
            rows.append(
                {
                    "period": period,
                    "fraction": fraction,
                    "rule": name,
                    "records": len(detections),
                    "reference_positive": int(positive.sum()),
                    "strict_misses": int(missed.sum()),
                    "positive_ties": int((positive & (flags == 5)).sum()),
                    "reference_negative_alerts": int((~positive & (flags >= 6)).sum()),
                    "positive_chemicals": detections.loc[positive, "conn"].nunique(),
                    "missed_chemicals": detections.loc[missed, "conn"].nunique(),
                }
            )
args.output.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(args.output, index=False)
