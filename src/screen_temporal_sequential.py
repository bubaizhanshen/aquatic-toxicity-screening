#!/usr/bin/env python3
"""Exploratory sequential testing on the reference-available Dutch panel.

Simulates an actual sequence: pick the next test using current predictions;
only then reveal that selected chemical's reference toxicity, and update scores.
Selection uses the historical year. Later-year observations are evaluated only
after all regional sequences are saved. This is retrospective model simulation.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--country", choices=["NL", "DE"], default="NL")
parser.add_argument("--classical", type=Path, required=True)
parser.add_argument("--neural", type=Path, required=True)
parser.add_argument(
    "--monitoring", type=Path, required=True, help="Reference-available panel"
)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--fixed-lists", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
opts = parser.parse_args()
OUT = opts.output
OUT.mkdir(parents=True, exist_ok=True)
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
PC = ["pred_" + m + "_p" for m in MODS]
KEY = ["site", "date", "time", "sampling_end_date"]
KS = (1, 3, 5, 10, 20)
FLU = "GVEPBJHOBDJJJI"
IND = "SXQBHARYMNFBPS"

c = pd.read_csv(opts.classical)
n = pd.read_csv(opts.neural)
p = c.merge(n[["conn", *PC[7:]]], on="conn", validate="one_to_one")
p = p[p.ad_inside_5nn][["conn", *PC]]
# Restrict eligibility to the previously defined panel with a reference available.
# Actual values are accessed only after the corresponding chemical is selected.
d = pd.read_csv(opts.monitoring, dtype={k: str for k in KEY}, low_memory=False).fillna(
    ""
)
d = (
    d.merge(p, on="conn", how="inner", validate="many_to_one")
    .sort_values(["area", "period", *KEY, "conn"])
    .reset_index(drop=True)
)
refs_table = pd.read_csv(opts.reference, usecols=["conn", "p_value"])
assert refs_table.conn.is_unique and refs_table.p_value.notna().all()
refs = refs_table.set_index("conn").p_value.to_dict()
fixed = json.loads(opts.fixed_lists.read_text())["regions"]
sequences = {}
for area, g in d[d.period.eq("selection")].groupby("area", sort=True):
    groups = [ev for _, ev in g.groupby(KEY, dropna=False, sort=True) if len(ev) >= 2]
    pool = sorted(g.conn.unique())
    if not groups or len(pool) < max(KS) or not all(x in refs for x in pool):
        raise ValueError(
            f"Insufficient selection events or candidate references in {area}"
        )
    width = max(len(ev) for ev in groups)
    ids = np.full((len(groups), width), -1, int)
    score = np.full((len(groups), width, 10), -np.inf)
    scale = np.zeros((len(groups), width))
    pos = {x: i for i, x in enumerate(pool)}
    for i, ev in enumerate(groups):
        L = len(ev)
        ids[i, :L] = [pos[x] for x in ev.conn]
        scale[i, :L] = (
            ev.concentration_ug_l.to_numpy(float) * 0.001 / ev.mw.to_numpy(float)
        )
        score[i, :L, :] = scale[i, :L, None] * 10 ** ev[PC].to_numpy(float)
    previous = []
    rows = []
    for iteration in range(1, max(KS) + 1):
        winners = score.argmax(axis=1)
        winnerids = ids[np.arange(len(ids))[:, None], winners]
        eligible = winnerids[~np.isin(winnerids, [pos[x] for x in previous])]
        freq = np.bincount(eligible, minlength=len(pool))
        available = [x for x in pool if x not in previous]
        choice = sorted(available, key=lambda x: (-freq[pos[x]], x))[0]
        # Only this selection's assay is now revealed to the sequential strategy.
        measured = refs[choice]
        previous.append(choice)
        m = ids == pos[choice]
        score[m, :] = np.repeat((scale[m] * 10**measured)[:, None], 10, axis=1)
        rows.append(
            {
                "iteration": iteration,
                "conn": choice,
                "unresolved_leader_votes_before_test": int(freq[pos[choice]]),
            }
        )
    assert previous[0] == fixed[area]["choices"]["leader5"][0]
    sequences[area] = {"pool_size": len(pool), "sequence": rows}

(OUT / "16_sequences_before_evaluation.json").write_text(
    json.dumps(
        {
            "retrospective": True,
            "rule": "Each round pick the most frequent untested model-event leader after replacing earlier tested chemicals with their reference values; lexical tie-break.",
            "selection_years": "2010–2018" if opts.country == "DE" else "2017",
            "evaluation_years": "2019–2023" if opts.country == "DE" else "2023",
            "regions": sequences,
        },
        indent=2,
    )
    + "\n"
)

results = []
events = []
for area, g in d[d.period.eq("evaluation")].groupby("area", sort=True):
    g = g.copy()
    g["ref"] = g.conn.map(refs)
    groups = [
        (key, ev)
        for key, ev in g.groupby(KEY, dropna=False, sort=True)
        if len(ev) >= 2 and ev.ref.notna().all()
    ]
    for key, ev in groups:
        scale = ev.concentration_ug_l.to_numpy(float) * 0.001 / ev.mw.to_numpy(float)
        true = scale * 10 ** ev.ref.to_numpy(float)
        score = scale[:, None] * 10 ** ev[PC].to_numpy(float)
        leader = ev.conn.iloc[int(true.argmax())]
        maxref = true.max()
        for k in KS:
            chosen = {x["conn"] for x in sequences[area]["sequence"][:k]}
            m = ev.conn.isin(chosen).to_numpy()
            updated = score.copy()
            updated[m, :] = true[m, None]
            winner = updated.argmax(axis=0)
            winref = true[winner]
            ratio = maxref / winref
            events.append(
                {
                    "area": area,
                    **dict(zip(KEY, key)),
                    "budget": k,
                    "leader": leader,
                    "no_indeno": IND not in set(ev.conn),
                    "top1": int((winref == maxref).sum() >= 5),
                    "loss10": int((ratio >= 10).sum() >= 5),
                    "loss3": int((ratio >= 3).sum() >= 5),
                    "top1_fraction": float((winref == maxref).mean()),
                    "regret": float((1 - winref / maxref).mean()),
                }
            )
e = pd.DataFrame(events)
e.to_csv(OUT / "16_evaluation_events.csv.gz", index=False, compression="gzip")
for k, sub in e.groupby("budget"):
    for group, mask in {
        "all": np.ones(len(sub), bool),
        "non_fluoranthene_leader": sub.leader.ne(FLU),
        "no_indeno": sub.no_indeno,
        "neither": sub.leader.ne(FLU) & sub.no_indeno,
    }.items():
        z = sub[mask]
        results.append(
            {
                "budget": int(k),
                "subgroup": group,
                "n": len(z),
                "top1": int(z.top1.sum()),
                "loss10": int(z.loss10.sum()),
                "loss3": int(z.loss3.sum()),
                "top1_fraction": float(z.top1_fraction.mean()),
                "regret": float(z.regret.mean()),
            }
        )
pd.DataFrame(results).to_csv(OUT / "16_results.csv", index=False)
print(
    json.dumps(
        {
            "events": len(e) // len(KS),
            "areas": len(sequences),
            "budgets": KS,
            "status": "complete",
        },
        indent=2,
    )
)
