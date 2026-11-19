#!/usr/bin/env python3
"""Evaluate fixed five-test strategies against 1,000 random lists per region."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np, pandas as pd

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
KEY = ["site", "date", "time", "sampling_end_date"]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--classical", type=Path, required=True)
parser.add_argument("--neural", type=Path, required=True)
parser.add_argument("--monitoring", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--exclude-events-containing", default="")
parser.add_argument("--expected-areas", type=int, default=5)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
c = pd.read_csv(args.classical)
n = pd.read_csv(args.neural)
assert c.conn.is_unique and n.conn.is_unique and set(c.conn) == set(n.conn)
p = c.merge(n[["conn", *PC[7:]]], on="conn", validate="one_to_one")
assert p[PC].notna().all().all()
p = p[p.ad_inside_5nn].copy()
d = pd.read_csv(
    args.monitoring,
    dtype={k: str for k in ["site", "date", "time", "sampling_end_date"]},
    low_memory=False,
).fillna("")
d = d.merge(p[["conn", *PC]], on="conn", how="inner", validate="many_to_one")
d = d.sort_values(["area", "period", *KEY, "conn"]).reset_index(drop=True)
assert not d.duplicated(["area", "period", *KEY, "conn"]).any()
# Freeze selections before joining any toxicity-reference value.
selections = {}
candidate_rows = []
for area, g in d[d.period.eq("selection")].groupby("area"):
    avail = sorted(g.conn.unique())
    assert len(avail) >= 5
    scale = g.concentration_ug_l.to_numpy(float) * 0.001 / g.mw.to_numpy(float)
    q = scale[:, None] * 10 ** g[PC].to_numpy(float)
    g = g.copy()
    g["peak_q"] = np.median(q, axis=1)
    peak = g.groupby("conn").peak_q.max().to_dict()
    leader = {x: 0 for x in avail}
    for _, ev in g.groupby(KEY, dropna=False, sort=True):
        if len(ev) < 2:
            continue
        arr = (
            ev.concentration_ug_l.to_numpy(float)[:, None]
            * 0.001
            / ev.mw.to_numpy(float)[:, None]
            * 10 ** ev[PC].to_numpy(float)
        )
        for i in arr.argmax(axis=0):
            leader[ev.conn.iloc[int(i)]] += 1
    order = lambda v: sorted(avail, key=lambda x: (-float(v.get(x, 0)), x))
    lead = order(leader)
    pk = order(peak)
    mixed = lead[:3] + [x for x in pk if x not in lead[:3]][:2]
    choices = {"none": [], "leader5": lead[:5], "peak5": pk[:5], "combined": mixed}
    selections[area] = {
        "candidate_pool": avail,
        "choices": choices,
        "selection_records": len(g),
    }
    for conn in avail:
        candidate_rows.append(
            {
                "area": area,
                "conn": conn,
                "leader_frequency": leader[conn],
                "peak_predicted_quotient": peak[conn],
                "selected_combined": conn in mixed,
            }
        )
assert len(selections) == args.expected_areas
(OUT / "04_frozen_selections.json").write_text(
    json.dumps(
        {
            "scope": "post-hoc; selection uses 2017 concentrations and fixed predictions, no toxicity references",
            "predictions_sha256": {
                "classical": hashlib.sha256(args.classical.read_bytes()).hexdigest(),
                "neural": hashlib.sha256(args.neural.read_bytes()).hexdigest(),
            },
            "regions": selections,
        },
        indent=2,
    )
    + "\n"
)
pd.DataFrame(candidate_rows).to_csv(OUT / "04_candidate_pool.csv", index=False)
# Reference labels are linked only after all selection lists are frozen.
labels = pd.read_csv(args.reference, usecols=["conn", "p_value"])
assert labels.conn.is_unique and labels.p_value.notna().all()
refmap = labels.set_index("conn").p_value.to_dict()
eval_data = d[d.period.eq("evaluation")].copy()
eval_data["reference_p"] = eval_data.conn.map(refmap)
if args.exclude_events_containing:
    bad = (
        eval_data[eval_data.conn.eq(args.exclude_events_containing)][["area", *KEY]]
        .drop_duplicates()
        .assign(_exclude=True)
    )
    eval_data = eval_data.merge(bad, on=["area", *KEY], how="left")
    eval_data = eval_data[eval_data["_exclude"].ne(True)].drop(columns="_exclude")
# Do not drop reference-missing chemicals within an event: exclude incomplete events instead.
blocks = {}
coverage = []
for area, g in eval_data.groupby("area"):
    groups = [(k, x) for k, x in g.groupby(KEY, dropna=False, sort=True) if len(x) >= 2]
    complete = [(k, x) for k, x in groups if x.reference_p.notna().all()]
    pool = selections[area]["candidate_pool"]
    assert len(pool) >= 5
    n_events = len(complete)
    width = max((len(x) for _, x in complete), default=0)
    assert n_events > 0
    universe = sorted(set(pool) | set(x for _, ev in complete for x in ev.conn))
    ix = {c: i for i, c in enumerate(universe)}
    ids = np.full((n_events, width), -1, int)
    ref = np.zeros((n_events, width))
    score = np.full((n_events, width, 10), -np.inf)
    keys = []
    for i, (key, ev) in enumerate(complete):
        nchem = len(ev)
        ids[i, :nchem] = [ix[x] for x in ev.conn]
        scale = ev.concentration_ug_l.to_numpy(float) * 0.001 / ev.mw.to_numpy(float)
        ref[i, :nchem] = scale * 10 ** ev.reference_p.to_numpy(float)
        score[i, :nchem, :] = scale[:, None] * 10 ** ev[PC].to_numpy(float)
        keys.append(key)
    assert np.all(ref.max(axis=1) > 0)
    blocks[area] = {
        "pool": pool,
        "universe": universe,
        "pos": ix,
        "ids": ids,
        "ref": ref,
        "score": score,
        "keys": keys,
    }
    coverage.append(
        {
            "area": area,
            "candidate_pool": len(pool),
            "candidate_reference_available": sum(c in refmap for c in pool),
            "evaluation_events_ge2_in_domain": len(groups),
            "evaluation_events_complete_reference": n_events,
            "excluded_incomplete_events": len(groups) - n_events,
            "evaluation_distinct_chemicals_complete": len(
                set(x for _, e in complete for x in e.conn)
            ),
            "sites_complete": len(set(k[0] for k in keys)),
            "max_chemicals_in_complete_event": width,
        }
    )


# Calculate decisions for all model/event pairs; missing padded scores never win.
def evaluate(b, chosen):
    mask = np.zeros(len(b["universe"]), bool)
    for c in chosen:
        if c in b["pos"]:
            mask[b["pos"][c]] = True
    ids = b["ids"]
    valid = ids >= 0
    selected = np.zeros_like(valid)
    selected[valid] = mask[ids[valid]]
    q = np.where(selected[:, :, None], b["ref"][:, :, None], b["score"])
    winner = q.argmax(axis=1)
    ref = b["ref"]
    wref = ref[np.arange(len(ref))[:, None], winner]
    maxref = ref.max(axis=1)
    ratio = maxref[:, None] / wref
    votes10 = (ratio >= 10).sum(axis=1)
    votes3 = (ratio >= 3).sum(axis=1)
    votes2 = (ratio >= 2).sum(axis=1)
    high = ref.sum(axis=1) >= 0.1
    sumq = np.where(np.isfinite(q), q, 0).sum(axis=1)
    miss = high & ((sumq < 0.1).sum(axis=1) >= 5)
    return {
        "loss10": votes10 >= 5,
        "loss3": votes3 >= 5,
        "loss2": votes2 >= 5,
        "high": high,
        "high_miss": miss,
        "regret": (1 - wref / maxref[:, None]).mean(axis=1),
        "top1_model_fraction": (wref == maxref[:, None]).mean(axis=1),
        "top1_majority": ((wref == maxref[:, None]).sum(axis=1) >= 5),
    }


summary = []
event_rows = []
base_by_area = {}
after_by_area = {}
for area, b in blocks.items():
    for strat, chosen in selections[area]["choices"].items():
        res = evaluate(b, chosen)
        if strat == "none":
            base_by_area[area] = res
        if strat == "combined":
            after_by_area[area] = res
        summary.append(
            {
                "area": area,
                "strategy": strat,
                "events": len(res["loss10"]),
                "priority_losses": int(res["loss10"].sum()),
                "losses_3fold": int(res["loss3"].sum()),
                "losses_2fold": int(res["loss2"].sum()),
                "reference_high_events": int(res["high"].sum()),
                "high_misses": int(res["high_miss"].sum()),
                "mean_regret": float(res["regret"].mean()),
                "mean_top1_model_fraction": float(res["top1_model_fraction"].mean()),
                "top1_majority_events": int(res["top1_majority"].sum()),
            }
        )
        for i, key in enumerate(b["keys"]):
            event_rows.append(
                {
                    "area": area,
                    "site": key[0],
                    "date": key[1],
                    "time": key[2],
                    "sampling_end_date": key[3],
                    "strategy": strat,
                    "priority_loss": bool(res["loss10"][i]),
                    "high_miss": bool(res["high_miss"][i]),
                    "reference_sum": float(b["ref"][i].sum()),
                    "mean_regret": float(res["regret"][i]),
                    "top1_model_fraction": float(res["top1_model_fraction"][i]),
                    "top1_majority": bool(res["top1_majority"][i]),
                }
            )
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "04_strategy_summary.csv", index=False)
pd.DataFrame(coverage).to_csv(OUT / "04_coverage.csv", index=False)
events = pd.DataFrame(event_rows)
events.to_csv(OUT / "04_event_results.csv.gz", index=False, compression="gzip")
rng = np.random.default_rng(20261115)
random = []
for rep in range(1000):
    row = {
        "replicate": rep,
        "total_priority_losses": 0,
        "total_high_misses": 0,
        "total_loss3": 0,
        "total_loss2": 0,
        "sum_regret": 0.0,
        "sum_top1_model_fraction": 0.0,
        "total_top1_majority": 0,
        "total_events": 0,
    }
    for area, b in blocks.items():
        chosen = rng.choice(b["pool"], size=5, replace=False).tolist()
        res = evaluate(b, chosen)
        loss = int(res["loss10"].sum())
        row[area] = loss
        row["total_priority_losses"] += loss
        row["total_high_misses"] += int(res["high_miss"].sum())
        row["total_loss3"] += int(res["loss3"].sum())
        row["total_loss2"] += int(res["loss2"].sum())
        row["sum_regret"] += float(res["regret"].sum())
        row["sum_top1_model_fraction"] += float(res["top1_model_fraction"].sum())
        row["total_top1_majority"] += int(res["top1_majority"].sum())
        row["total_events"] += len(res["loss10"])
    random.append(row)
rnd = pd.DataFrame(random)
rnd["mean_regret"] = rnd.sum_regret / rnd.total_events
rnd["mean_top1_model_fraction"] = rnd.sum_top1_model_fraction / rnd.total_events
rnd.to_csv(OUT / "04_random_1000.csv.gz", index=False, compression="gzip")
b = summary[summary.strategy.eq("none")]
a = summary[summary.strategy.eq("combined")]
base = int(b.priority_losses.sum())
after = int(a.priority_losses.sum())
paired = []
for area, bv in base_by_area.items():
    av = after_by_area[area]
    keys = blocks[area]["keys"]
    paired += [
        {"area": area, "site": k[0], "delta": int(x) - int(y)}
        for k, x, y in zip(keys, bv["loss10"], av["loss10"])
    ]
site = pd.DataFrame(paired).groupby(["area", "site"], as_index=False).delta.sum()
rng = np.random.default_rng(21561115)
boot = []
for _ in range(2000):
    total = 0
    for area in blocks:
        z = site[site.area.eq(area)].delta.to_numpy()
        total += int(rng.choice(z, size=len(z), replace=True).sum())
    boot.append(total)
by_region = []
for area in blocks:
    aa = int(a[a.area.eq(area)].priority_losses.iloc[0])
    v = rnd[area].to_numpy()
    by_region.append(
        {
            "area": area,
            "random_median": float(np.median(v)),
            "random_at_least_as_good": float((v <= aa).mean()),
            "random_strictly_better": float((v < aa).mean()),
            "method_after": aa,
        }
    )
payload = {
    "status": "completed fitted-model retrospective evaluation",
    "events": int(b.events.sum()),
    "before_priority_losses": base,
    "after_priority_losses": after,
    "relative_reduction": (base - after) / base if base else None,
    "random_median": float(rnd.total_priority_losses.median()),
    "random_at_least_as_good": float((rnd.total_priority_losses <= after).mean()),
    "random_strictly_better": float((rnd.total_priority_losses < after).mean()),
    "random_quantiles": np.quantile(
        rnd.total_priority_losses, [0.025, 0.5, 0.975]
    ).tolist(),
    "conditional_site_bootstrap_net_reduction": np.quantile(
        boot, [0.025, 0.975]
    ).tolist(),
    "before_high_sum_events": int(b.reference_high_events.sum()),
    "before_high_misses": int(b.high_misses.sum()),
    "after_high_misses": int(a.high_misses.sum()),
    "random_by_region": by_region,
    "coverage": coverage,
    "excluded_events_containing": args.exclude_events_containing or None,
    "note": "Joint chemical holdout; retrospective and prior-domain-filtered; complete references for all matched in-domain chemicals within each eligible event. Any event exclusion is post hoc.",
}
(OUT / "04_results.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
