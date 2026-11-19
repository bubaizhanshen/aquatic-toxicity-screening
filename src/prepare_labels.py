#!/usr/bin/env python3
"""Aggregate source records by structure and assign shared cross-validation folds."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

FEATURES = {
    "phenol": "[c]-[O;X2;H1]",
    "aliphatic_alcohol": "[C;X4]-[O;X2;H1]",
    "aromatic_amine": "[c]-[N;X3;+0;!$([N][C](=O));!$([N][S](=O));!$([N][P](=O))]",
    "amine_n_sp3_only": "[N;X3;+0;!$([N]~[!#6;!#1]);!$([N]~[c]);!$([N]~[C;!X4])]-[C;X4]",
    "aromatic_halogen": "[c]-[F,Cl,Br,I]",
    "aliphatic_halogen": "[C;X4]-[F,Cl,Br,I]",
    "aromatic_nitro": "[c]-[N+](=O)[O-]",
    "carboxylic_acid": "[C;X3](=O)[O;X2;H1]",
    "carboxylic_ester": "[C;X3](=O)[O;X2][#6]",
    "organophosphorus": "[#15]~[#6,#7,#8,#16]",
}


def add_fragments(frame):
    molecules = [Chem.MolFromSmiles(smiles) for smiles in frame.smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Invalid SMILES")
    for name, smarts in FEATURES.items():
        pattern = Chem.MolFromSmarts(smarts)
        frame[f"fragment_{name}"] = [
            mol.HasSubstructMatch(pattern) for mol in molecules
        ]
    return frame


def aggregate(records, source):
    if source == "broad":
        eligible = (
            records.conn.notna()
            & np.isfinite(records.p_value)
            & records.mg_l.gt(0)
            & np.isfinite(records.p_minus_converted_p)
            & records.p_minus_converted_p.abs().le(0.05)
        )
        records = records[eligible]
    else:
        group = ["endpoint", "conn", "database", "source_reference"]
        required = group + ["smiles", "p_value"]
        records = records.dropna(subset=required)
        records = records[np.isfinite(records.p_value)].copy()
        records["rounded_value"] = records.p_value.round(6)
        records = records.drop_duplicates(group + ["rounded_value"])
        records = records.groupby(group, as_index=False, sort=False).agg(
            smiles=("smiles", "first"),
            p_value=("p_value", "median"),
        )
    result = records.groupby(["endpoint", "conn"], as_index=False).agg(
        smiles=("smiles", "first"),
        p_value=("p_value", "median"),
    )
    result["fold"] = result.conn.map(
        lambda conn: int(hashlib.sha256(conn.encode()).hexdigest(), 16) % 5
    )
    return add_fragments(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--source", choices=["broad", "study"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = pd.read_csv(args.records, low_memory=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregate(records, args.source).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
