#!/usr/bin/env python3
"""Replay California detection screening and EPA benchmark comparisons."""
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--monitoring", type=Path, required=True)
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--source-ranges", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
WORK = args.output
WORK.mkdir(parents=True, exist_ok=True)
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
PC = [f"pred_{m}_p" for m in MODS]
original = pd.read_csv(args.monitoring, dtype={"cas_key": str}, low_memory=False)
d = original[
    (original.species == "Daphnia magna")
    & (original.cohort == "original_chemical_fold")
    & (~original.stereo_four_sensitivity_exclusion)
    & original.has_reference
].copy()
assert len(d) == 3940 and d.conn.nunique() == 72
fit = pd.read_csv(args.predictions).set_index("conn")
assert fit.index.is_unique and set(d.conn) <= set(fit.index)
d.loc[:, PC] = fit.loc[d.conn, PC].to_numpy()
d["p_value"] = fit.loc[d.conn, "p_value"].to_numpy()
scale = d.concentration_ug_l.to_numpy(float) * 0.001 / d.mw.to_numpy(float)
qp = scale[:, None] * 10 ** d[PC].to_numpy(float)
sources = pd.read_csv(args.source_ranges).set_index("conn")
assert set(d.conn) <= set(sources.index)
summary = []
detail = []
for fraction in (0.5, 0.1, 0.01, 0.001):
    for scenario, values in (
        ("central", d.p_value.to_numpy(float)),
        ("least_potent_source_median", sources.loc[d.conn, "p_min"].to_numpy(float)),
        ("most_potent_source_median", sources.loc[d.conn, "p_max"].to_numpy(float)),
    ):
        qr = scale * 10**values
        positive = (qr >= fraction) | np.isclose(qr, fraction, rtol=1e-12, atol=0)
        vote = ((qp >= fraction) | np.isclose(qp, fraction, rtol=1e-12, atol=0)).sum(
            axis=1
        )
        miss = positive & (vote < 5)
        tie = positive & (vote == 5)
        flag = positive & (vote >= 6)
        epa = d.epa_exceeded.to_numpy(bool)
        both = epa & positive
        summary.append(
            {
                "fraction": fraction,
                "scenario": scenario,
                "records": len(d),
                "reference_positive": int(positive.sum()),
                "positive_chemicals": int(d.loc[positive, "conn"].nunique()),
                "strict_misses": int(miss.sum()),
                "missed_chemicals": int(d.loc[miss, "conn"].nunique()),
                "missed_sites": int(d.loc[miss, "StationCode"].nunique()),
                "ties": int(tie.sum()),
                "flags": int(flag.sum()),
                "epa_positive": int(epa.sum()),
                "epa_and_reference_positive": int(both.sum()),
                "epa_and_reference_positive_and_missed": int((both & miss).sum()),
                "epa_positive_model_missed": int((epa & (vote < 5)).sum()),
                "epa_positive_model_missed_reference_negative": int(
                    (epa & (vote < 5) & ~positive).sum()
                ),
            }
        )
        if scenario == "central":
            d[f"reference_positive_{fraction}"] = positive
            d[f"votes_{fraction}"] = vote
            d[f"miss_{fraction}"] = miss
            d[f"tie_{fraction}"] = tie
        if fraction == 0.1:
            for c, g in d.groupby("conn"):
                ix = d.conn.eq(c).to_numpy()
                detail.append(
                    {
                        "fraction": fraction,
                        "scenario": scenario,
                        "conn": c,
                        "name": g.DW_AnalyteName.iloc[0],
                        "detections": int(ix.sum()),
                        "reference_positive": int((positive & ix).sum()),
                        "misses": int((miss & ix).sum()),
                        "missed_sites": int(d.loc[miss & ix, "StationCode"].nunique()),
                        "source_study_count": (
                            int(g.conn.map(sources["n_studies"]).iloc[0])
                            if "n_studies" in sources
                            else None
                        ),
                    }
                )
summary = pd.DataFrame(summary)
detail = pd.DataFrame(detail)
summary.to_csv(WORK / "corrected_ceden_summary.csv", index=False)
detail.to_csv(WORK / "corrected_ceden_by_chemical.csv", index=False)
d.to_csv(WORK / "corrected_ceden_detections.csv.gz", index=False, compression="gzip")

pos = d["reference_positive_0.1"].to_numpy(bool)
votes = d["votes_0.1"].to_numpy(int)
vote_rows = []
for minimum in (1, 6):
    flagged = votes >= minimum
    vote_rows.append(
        {
            "minimum_model_flags": minimum,
            "positive_flagged": int((pos & flagged).sum()),
            "positive_unflagged": int((pos & ~flagged).sum()),
            "negative_flagged": int((~pos & flagged).sum()),
            "negative_unflagged": int((~pos & ~flagged).sum()),
        }
    )
pd.DataFrame(vote_rows).to_csv(WORK / "corrected_ceden_vote_workload.csv", index=False)
main = summary[(summary.scenario == "central") & summary.fraction.isin([0.1, 0.5])]
payload = {
    "main": main.to_dict("records"),
    "votes": vote_rows,
    "source_extremes_0_1": summary[
        (summary.fraction == 0.1) & (summary.scenario != "central")
    ].to_dict("records"),
    "note": "Reference scenarios use measured study medians rather than confidence limits.",
}
(WORK / "corrected_ceden_key_results.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
print(json.dumps(payload, indent=2))
