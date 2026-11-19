#!/usr/bin/env python3
"""Morgan inputs, training-derived applicability domain, and primary model settings."""
import warnings

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy import sparse

SEED = 20261103
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
DESCRIPTORS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHAcceptors,
    Descriptors.NumHDonors,
    Descriptors.NumRotatableBonds,
    Descriptors.RingCount,
]


def get_data(x):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in x.smiles]
    if any(m is None for m in mols):
        raise ValueError("Invalid SMILES after freeze")
    fp = [generator.GetFingerprint(m) for m in mols]
    bits = np.vstack([generator.GetFingerprintAsNumPy(m) for m in mols]).astype(
        np.float32
    )
    desc = np.asarray([[float(fn(m)) for fn in DESCRIPTORS] for m in mols], dtype=float)
    if not np.isfinite(desc).all():
        raise ValueError("Nonfinite descriptor")
    return fp, sparse.csr_matrix(bits), desc


def top5(fp, train_indices, query_indices, is_training=False):
    ref = [fp[i] for i in train_indices]
    means, nearest, neighbor_indices = [], [], []
    # Remove each molecule itself when building the training cutoff.
    positions = {idx: k for k, idx in enumerate(train_indices)}
    for i in query_indices:
        sim = np.asarray(DataStructs.BulkTanimotoSimilarity(fp[i], ref), dtype=float)
        if is_training:
            sim[positions[i]] = -1.0
        idx = np.argpartition(sim, -5)[-5:]
        means.append(float(sim[idx].mean()))
        nearest.append(float(sim[idx].max()))
        neighbor_indices.append(train_indices[idx])
    return np.asarray(means), np.asarray(nearest), np.asarray(neighbor_indices)


def models():
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    return {
        "random_forest": RandomForestRegressor(
            n_estimators=180,
            n_jobs=1,
            random_state=SEED,
            max_features="sqrt",
            min_samples_leaf=1,
        ),
        "lightgbm": LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=1,
            verbosity=-1,
        ),
        "xgboost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1,
            tree_method="hist",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        ),
        "svr": SVR(kernel="rbf", C=1.0, epsilon=0.15, gamma="scale", cache_size=256),
    }
