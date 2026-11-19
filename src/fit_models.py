#!/usr/bin/env python3
"""Fit the seven conventional and three neural models on curated toxicity records."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from molecular_models import SEED, get_data, models, top5
from prepare_labels import add_fragments


def read_structures(path, with_labels):
    columns = ["conn", "smiles", "p_value"] if with_labels else ["conn", "smiles"]
    frame = (
        pd.read_csv(path, usecols=columns).sort_values("conn").reset_index(drop=True)
    )
    if not frame.conn.is_unique or frame[columns].isna().any().any():
        raise ValueError(f"Expected one complete record per chemical in {path}")
    if with_labels and not np.isfinite(frame.p_value).all():
        raise ValueError(f"Non-finite reference toxicity in {path}")
    return frame


def fold_for(conn):
    return int(hashlib.sha256(conn.encode()).hexdigest(), 16) % 5


def classical(bits, descriptors, truth, train, test, offset, legacy_descriptors):
    from catboost import CatBoostRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    primary = models()
    expanded = {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=1,
            random_state=SEED + offset,
        ),
        "ridge": Ridge(alpha=10.0, solver="lsqr"),
        "catboost": CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            thread_count=1,
            random_seed=SEED + offset,
            verbose=False,
            allow_writing_files=False,
        ),
    }
    design = sparse.hstack(
        [
            bits,
            sparse.csr_matrix(
                StandardScaler().fit(descriptors[train]).transform(descriptors)
            ),
        ],
        format="csr",
    )
    extra_desc = descriptors.astype(np.float32) if legacy_descriptors else descriptors
    extra_design = sparse.hstack(
        [
            bits,
            sparse.csr_matrix(
                StandardScaler().fit(extra_desc[train]).transform(extra_desc)
            ),
        ],
        format="csr",
    )
    result = {}
    for suite, inputs, labels in (
        (primary, design, truth),
        (expanded, extra_design, truth.astype(np.float32).astype(float)),
    ):
        for name, estimator in suite.items():
            estimator.fit(inputs[train], labels[train])
            result[f"pred_{name}_p"] = estimator.predict(inputs[test])
    return result


def neural(graphs, descriptors, truth, train, test, offset, engine, epoch_limits):
    result = {}
    metadata = []
    for index, name in enumerate(("mlp", "gine", "gat")):
        for limit in epoch_limits:
            engine.MAX_EPOCHS = limit
            prediction, details = engine.model_fit(
                name,
                graphs,
                train,
                test,
                truth.astype(np.float32),
                descriptors,
                SEED + offset + 100 * index,
            )
            if details["epochs_ran"] - details["best_epoch"] >= engine.PATIENCE:
                break
        else:
            raise RuntimeError(f"No complete early stopping window for {name}")
        result[f"pred_{name}_p"] = prediction
        details["maximum_epochs"] = limit
        metadata.append(details)
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--query", type=Path, help="Separately held-out chemical identities"
    )
    parser.add_argument(
        "--fold", type=int, choices=range(5), help="Fit one of the five CV folds"
    )
    parser.add_argument(
        "--seed-offset", type=int, help="Seed offset for a joint or external holdout"
    )
    parser.add_argument(
        "--component", choices=("classical", "neural", "all"), default="all"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--legacy-descriptors",
        action="store_true",
        help="Scale float32 descriptors for the initial three expanded conventional fits",
    )
    parser.add_argument("--epoch-limits", type=int, nargs="+", default=(100, 300, 600))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.query and args.fold is not None:
        parser.error("--query and --fold are exclusive")
    if args.seed_offset is not None and not args.query:
        parser.error("--seed-offset requires --query")

    known = read_structures(args.train, with_labels=True)
    if args.query:
        query = read_structures(args.query, with_labels=False)
        if set(query.conn) & set(known.conn):
            raise ValueError("A query connectivity is present in training")
        data = pd.concat([known, query], ignore_index=True)
        splits = [
            (
                args.seed_offset if args.seed_offset is not None else 5,
                np.arange(len(known)),
                np.arange(len(known), len(data)),
            )
        ]
    else:
        data = known
        folds = data.conn.map(fold_for).to_numpy()
        selected = range(5) if args.fold is None else [args.fold]
        splits = [
            (fold, np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
            for fold in selected
        ]

    if any(len(train) < 6 or not len(test) for _, train, test in splits):
        raise ValueError(
            "Each split needs at least six training chemicals and a nonempty holdout"
        )
    fp, bits, desc = get_data(data)
    target = data.p_value.to_numpy(float)
    nn_desc = desc.astype(np.float32)
    if args.component in ("neural", "all"):
        import torch
        import neural_models as engine

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no compatible GPU is available")
        engine.DEVICE = torch.device(args.device)
        dense = bits.toarray().astype(np.uint8)
        graphs = [
            engine.graph(smiles, dense[i], nn_desc[i], i)
            for i, smiles in enumerate(data.smiles)
        ]

    results = []
    metadata = []
    for offset, train, test in splits:
        if len(set(data.conn.iloc[train]) & set(data.conn.iloc[test])):
            raise ValueError("A chemical is shared between training and holdout")
        prediction = data.iloc[test][["conn", "smiles"]].copy()
        if not args.query:
            prediction["fold"] = offset
            prediction["p_value"] = data.p_value.iloc[test].to_numpy()
        if args.component in ("classical", "all"):
            mean_train, _, _ = top5(fp, train, train, is_training=True)
            mean, nearest, neighbors = top5(fp, train, test)
            cutoff = np.quantile(mean_train, 0.05)
            prediction["ad_mean_5nn_similarity"] = mean
            prediction["ad_nearest_similarity"] = nearest
            prediction["ad_training_5pct_cutoff"] = cutoff
            prediction["ad_inside_5nn"] = mean >= cutoff
            prediction["pred_5nn_readacross_p"] = np.median(target[neighbors], axis=1)
            for key, value in classical(
                bits, desc, target, train, test, offset, args.legacy_descriptors
            ).items():
                prediction[key] = value
        if args.component in ("neural", "all"):
            values, details = neural(
                graphs, nn_desc, target, train, test, offset, engine, args.epoch_limits
            )
            for key, value in values.items():
                prediction[key] = value
            metadata.extend(details)
        results.append(prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat(results, ignore_index=True)
    if not args.query and args.component != "neural":
        table = add_fragments(table)
    table.to_csv(args.output, index=False)
    if metadata:
        args.output.with_suffix(".training.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
