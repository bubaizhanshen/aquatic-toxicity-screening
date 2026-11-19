#!/usr/bin/env python3
"""Replay U.S. event rankings, test allocation rules, and their sensitivity analyses."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--monitoring", type=Path, required=True)
parser.add_argument(
    "--predictions",
    type=Path,
    required=True,
    help="Predictions for the 52 jointly held-out monitoring chemicals",
)
parser.add_argument(
    "--reference",
    type=Path,
    required=True,
    help="Corrected reference table with conn and p_value",
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
OUT = args.output
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
PRED = ["pred_" + m + "_p" for m in MODS]
KEY = ["region", "SITE_NO", "YYYYMMDD", "TIME"]
print("Constructing frozen 52-chemical event matrices", flush=True)
p = pd.read_csv(args.predictions).set_index("conn")
assert len(p) == 52 and p.ad_inside_5nn.all()
pool = sorted(p.index)
pos = {c: i for i, c in enumerate(pool)}
n = len(pool)
m = len(MODS)
raw = pd.read_csv(args.monitoring, dtype={k: str for k in KEY[1:]}, low_memory=False)
raw = raw[raw.ad_inside_5nn & raw.conn.isin(pool)].copy()
reference = (
    pd.read_csv(args.reference, usecols=["conn", "p_value"]).set_index("conn").p_value
)
assert reference.index.is_unique and set(raw.conn).issubset(reference.index)
raw["p_value"] = raw.conn.map(reference)
assert raw.p_value.notna().all()
raw = raw[raw.groupby(KEY).conn.transform("size").ge(2)].sort_values(KEY + ["conn"])
assert not raw.duplicated(KEY + ["conn"]).any()
groups = list(raw.groupby(KEY, sort=True))
assert len(groups) == 2565 and set(raw.conn) == set(pool)
N = len(groups)
ref = np.zeros((N, n))
pred = np.zeros((N, n, m))
present = np.zeros((N, n), dtype=bool)
keys = []
est = []
names = {}
observed_ref = {}
flags = {}
for i, (key, g) in enumerate(groups):
    keys.append(key)
    est.append(bool(g.estimated.any()))
    ids = np.array([pos[c] for c in g.conn])
    scale = g.concentration_ng_l.to_numpy(float) * 1e-6 / g.mw.to_numpy(float)
    ref[i, ids] = scale * 10.0 ** g.p_value.to_numpy(float)
    pred[i, ids, :] = scale[:, None] * 10.0 ** p.loc[g.conn, PRED].to_numpy(float)
    present[i, ids] = True
    names.update(dict(zip(g.conn, g.chemical_names)))
    for conn, lab, flag in zip(g.conn, g.p_value, g.fragment_organophosphorus):
        if conn in observed_ref:
            assert np.isclose(observed_ref[conn], lab, rtol=0, atol=1e-10)
        if conn in flags:
            assert flags[conn] == bool(flag)
        observed_ref[conn] = float(lab)
        flags[conn] = bool(flag)
keys = pd.DataFrame(keys, columns=KEY)
estimated = np.array(est, dtype=bool)
regions = sorted(keys.region.unique())
assert len(regions) == 5
op = np.array([flags[c] for c in pool])
names = [names[c] for c in pool]


class Evaluate:
    def __init__(self, ix):
        self.ix = np.asarray(ix)
        self.r = ref[self.ix]
        self.p = pred[self.ix]
        self.k = len(self.ix)
        self.order = np.argsort(-self.p, axis=1, kind="stable")
        self.ar = np.arange(self.k)[:, None]
        self.mr = np.arange(m)[None, :]
        self.target = self.r.max(axis=1)
        self.high = self.r.sum(axis=1) >= 0.1

    def __call__(self, selection):
        selected = np.zeros(n, dtype=bool)
        selected[list(selection)] = True
        chosen = np.flatnonzero(selected)
        if len(chosen) == n:
            winner = np.broadcast_to(self.r.argmax(axis=1)[:, None], (self.k, m))
        else:
            order = self.order[:, : len(chosen) + 1, :]
            first = (~selected[order]).argmax(axis=1)
            best_pred = np.take_along_axis(order, first[:, None, :], axis=1)[:, 0, :]
            if len(chosen):
                chosen_ref = self.r[:, chosen]
                best_ref = chosen[chosen_ref.argmax(axis=1)]
                best_score = self.r[np.arange(self.k), best_ref][:, None]
                old_score = self.p[self.ar, best_pred, self.mr]
                use = (best_score > old_score) | (
                    (best_score == old_score) & (best_ref[:, None] < best_pred)
                )
                winner = np.where(use, best_ref[:, None], best_pred)
            else:
                winner = best_pred
        winner_ref = self.r[self.ar, winner]
        loss = (10 * winner_ref <= self.target[:, None]).sum(axis=1) >= 5
        adjusted_sum = self.p.sum(axis=1)
        if len(chosen):
            adjusted_sum = (
                adjusted_sum
                - self.p[:, chosen, :].sum(axis=1)
                + self.r[:, chosen].sum(axis=1)[:, None]
            )
        missed = self.high & ((adjusted_sum < 0.1).sum(axis=1) >= 5)
        return loss, missed, winner


def rank_for_region(region, eligible):
    train = eligible & keys.region.ne(region).to_numpy()
    pt = pred[train]
    avail = present[train].any(axis=0)
    freq = np.bincount(np.argmax(pt, axis=1).reshape(-1), minlength=n)
    peak = np.max(np.median(pt, axis=2), axis=0)
    ordered = lambda vals: sorted(
        np.flatnonzero(avail), key=lambda j: (-float(vals[j]), pool[j])
    )
    return ordered(freq), ordered(peak), np.flatnonzero(avail), freq, peak


def select_mix(leader, peak, lead, budget):
    selected = list(leader[:lead])
    for j in peak:
        if len(selected) >= budget:
            break
        if j not in selected:
            selected.append(j)
    assert len(selected) == budget and len(set(selected)) == budget
    return selected


evaluators = {r: Evaluate(np.flatnonzero(keys.region.eq(r))) for r in regions}
baseline = np.zeros(N, dtype=bool)
before_high = np.zeros(N, dtype=bool)
after = np.zeros(N, dtype=bool)
strategies = {str(k) + "+" + str(5 - k): np.zeros(N, dtype=bool) for k in range(6)}
strategy_high = {name: np.zeros(N, dtype=bool) for name in strategies}
selected_by_region = {}
rankings = {}
details = []
cumulative = []
counter = {
    name: np.zeros(N, dtype=bool)
    for name in ("correct_organophosphorus", "correct_other")
}
for region in regions:
    ev = evaluators[region]
    leader, peak, avail, freq, peakscore = rank_for_region(
        region, np.ones(N, dtype=bool)
    )
    rankings[region] = (leader, peak, avail)
    baseline[ev.ix], before_high[ev.ix], _ = ev([])
    for k in range(6):
        name = f"{k}+{5-k}"
        selected = select_mix(leader, peak, k, 5)
        strategies[name][ev.ix], strategy_high[name][ev.ix], _ = ev(selected)
    chosen = select_mix(leader, peak, 3, 5)
    after[ev.ix] = strategies["3+2"][ev.ix]
    selected_by_region[region] = [pool[j] for j in chosen]
    for name, sel in [
        ("correct_organophosphorus", np.flatnonzero(op)),
        ("correct_other", np.flatnonzero(~op)),
    ]:
        counter[name][ev.ix], _, _ = ev(sel)
    for step, j in enumerate(chosen, 1):
        single, _, _ = ev([j])
        cum, _, _ = ev(chosen[:step])
        fold = 10.0 ** (observed_ref[pool[j]] - p.loc[pool[j], PRED].to_numpy(float))
        details.append(
            {
                "region": region,
                "order": step,
                "selection_basis": (
                    "leader_frequency" if step <= 3 else "predicted_quotient_peak"
                ),
                "conn": pool[j],
                "name": names[j],
                "organophosphorus": bool(op[j]),
                "event_occurrences": int(present[ev.ix, j].sum()),
                "median_predicted_ec50_over_reference": float(np.median(fold)),
                "single_original_losses_corrected": int(
                    np.sum(baseline[ev.ix] & ~single)
                ),
                "single_new_losses": int(np.sum(~baseline[ev.ix] & single)),
                "single_net_reduction": int(baseline[ev.ix].sum() - single.sum()),
            }
        )
        cumulative.append(
            {
                "region": region,
                "step": step,
                "name": names[j],
                "remaining_losses": int(cum.sum()),
                "net_reduction": int(baseline[ev.ix].sum() - cum.sum()),
            }
        )

event = keys.copy()
event["estimated"] = estimated
event["baseline_loss"] = baseline
event["mixed_3_plus_2_loss"] = after
event["reference_high"] = ref.sum(axis=1) >= 0.1
event["baseline_high_miss"] = before_high
for name, values in strategies.items():
    event["loss_" + name] = values
    event["high_miss_" + name] = strategy_high[name]
for name, values in counter.items():
    event[name] = values
event["any_OP_detected"] = present[:, op].any(axis=1)
event.to_csv(OUT / "region_event_comparisons.csv.gz", index=False, compression="gzip")
pd.DataFrame(details).to_csv(OUT / "region_selected_chemical_effects.csv", index=False)
pd.DataFrame(cumulative).to_csv(OUT / "region_cumulative_effects.csv", index=False)

summary = []
for region in regions + ["All"]:
    ix = np.arange(N) if region == "All" else np.flatnonzero(keys.region.eq(region))
    for name, z in [("none", baseline), *strategies.items(), *counter.items()]:
        v = z[ix]
        b = baseline[ix]
        summary.append(
            {
                "region": region,
                "scenario": name,
                "events": len(ix),
                "priority_losses": int(v.sum()),
                "baseline_losses_corrected": int((b & ~v).sum()),
                "new_losses": int((~b & v).sum()),
                "net_reduction": int(b.sum() - v.sum()),
            }
        )
pd.DataFrame(summary).to_csv(OUT / "region_scenarios.csv", index=False)

# Removing whole events containing specified chemicals retains valid sample ranking comparisons.
filters = {
    "no_estimates": ~estimated,
    "without_chlorpyrifos_events": ~present[:, pool.index("SBPBAQFWLVIOKP")],
    "without_chlorpyrifos_or_malathion_events": ~present[
        :, pool.index("SBPBAQFWLVIOKP")
    ]
    & ~present[:, pool.index("JXSJBGJIGXNWCI")],
    "without_carbendazim_events": ~present[:, pool.index("TWFZGCMQGLPBSX")],
}
sens = []
for name, mask in filters.items():
    b = baseline[mask]
    v = after[mask]
    sens.append(
        {
            "subset": name,
            "events": int(mask.sum()),
            "baseline_loss": int(b.sum()),
            "fixed_original_mix_loss": int(v.sum()),
            "corrected": int((b & ~v).sum()),
            "new_losses": int((~b & v).sum()),
            "net": int(b.sum() - v.sum()),
        }
    )
# Reselecting after filtering corresponds to the pre-existing no-estimate analysis.
reselected = np.zeros(N, dtype=bool)
for region in regions:
    mask = (~estimated) & keys.region.eq(region).to_numpy()
    ev = Evaluate(np.flatnonzero(mask))
    leader, peak, _, _, _ = rank_for_region(region, ~estimated)
    chosen = select_mix(leader, peak, 3, 5)
    reselected[ev.ix], _, _ = ev(chosen)
print("reselected no-estimate losses", int(reselected.sum()), flush=True)
sens.append(
    {
        "subset": "no_estimates_reselect_in_other_regions",
        "events": int((~estimated).sum()),
        "baseline_loss": int(baseline[~estimated].sum()),
        "fixed_original_mix_loss": int(reselected.sum()),
        "corrected": int((baseline[~estimated] & ~reselected[~estimated]).sum()),
        "new_losses": int((~baseline[~estimated] & reselected[~estimated]).sum()),
        "net": int(baseline[~estimated].sum() - reselected.sum()),
    }
)
pd.DataFrame(sens).to_csv(OUT / "region_event_subset_sensitivity.csv", index=False)

# Single-chemical replacements are evaluated separately; their effects cannot
# be summed because the predicted winner can change after other replacements.
all_chem = []
reference_winner = ref.argmax(axis=1)
predicted_winner = pred.argmax(axis=1)
for j, conn in enumerate(pool):
    adjusted = np.zeros(N, dtype=bool)
    for region in regions:
        ev = evaluators[region]
        adjusted[ev.ix], _, _ = ev([j])
    ratio = 10.0 ** (observed_ref[conn] - p.loc[conn, PRED].to_numpy(float))
    all_chem.append(
        {
            "conn": conn,
            "name": names[j],
            "organophosphorus": bool(op[j]),
            "detected_events": int(present[:, j].sum()),
            "baseline_loss_events_with_detection": int(
                (baseline & present[:, j]).sum()
            ),
            "reference_leader_in_baseline_losses": int(
                (baseline & (reference_winner == j)).sum()
            ),
            "model_winner_counts_in_baseline_losses": int(
                (predicted_winner[baseline] == j).sum()
            ),
            "median_predicted_ec50_over_reference": float(np.median(ratio)),
            "single_corrected_original_losses": int((baseline & ~adjusted).sum()),
            "single_new_losses": int((~baseline & adjusted).sum()),
            "single_net_reduction": int(baseline.sum() - adjusted.sum()),
        }
    )
pd.DataFrame(all_chem).sort_values("single_net_reduction", ascending=False).to_csv(
    OUT / "region_all_chemical_individual_effects.csv", index=False
)

# Evaluate top-k identification and continuous decision regret at each model x
# event; at-least-five counts retain the original manuscript's voting unit.
adjusted = pred.copy()
for region in regions:
    ix = evaluators[region].ix
    chosen = [pos[c] for c in selected_by_region[region]]
    adjusted[np.ix_(ix, chosen, np.arange(m))] = ref[np.ix_(ix, chosen)][:, :, None]
top_rows = []
top_eligible = []
for label, qq in [("baseline", pred), ("after_3_plus_2", adjusted)]:
    order = np.argsort(-qq, axis=1, kind="stable")
    winner = order[:, 0, :]
    winner_ref = ref[np.arange(N)[:, None], winner]
    regret = 1 - winner_ref / ref.max(axis=1)[:, None]
    for k in [1, 3, 5]:
        hit = (order[:, :k, :] == reference_winner[:, None, None]).any(axis=1)
        top_rows.append(
            {
                "scenario": label,
                "top_k": k,
                "model_event_hits": int(hit.sum()),
                "model_event_total": N * m,
                "events_with_at_least_five_models_hitting": int(
                    (hit.sum(axis=1) >= 5).sum()
                ),
                "events": N,
                "mean_model_event_regret": float(regret.mean()),
                "median_model_event_regret": float(np.median(regret)),
            }
        )
        eligible = present.sum(axis=1) >= k
        top_eligible.append(
            {
                "scenario": label,
                "top_k": k,
                "events_with_at_least_k_chemicals": int(eligible.sum()),
                "events_with_at_least_five_models_hitting": int(
                    (hit[eligible].sum(axis=1) >= 5).sum()
                ),
                "model_event_hits": int(hit[eligible].sum()),
                "model_event_total": int(eligible.sum()) * m,
            }
        )
pd.DataFrame(top_rows).to_csv(OUT / "region_top_k_and_regret.csv", index=False)
pd.DataFrame(top_eligible).to_csv(OUT / "region_top_k_eligible_events.csv", index=False)

# Budget curves evaluate the two prespecified rankers; selection is by the other
# four regions for each target. These are retrospective comparisons.
budget_rows = []
for b in [0, 1, 2, 3, 4, 5, 10]:
    for mode in ("leader_frequency", "peak_quotient"):
        total = 0
        high = 0
        for region in regions:
            ev = evaluators[region]
            leader, peak, _ = rankings[region]
            chosen = (leader if mode == "leader_frequency" else peak)[:b]
            l, h, _ = ev(chosen)
            total += int(l.sum())
            high += int(h.sum())
        budget_rows.append(
            {
                "budget_per_region": b,
                "strategy": mode,
                "priority_losses": total,
                "high_quotient_misses": high,
            }
        )
pd.DataFrame(budget_rows).to_csv(OUT / "region_budget_curves.csv", index=False)

# Random sets are drawn independently from each region's other-region candidate
# pool. The spread is selection variability conditional on the observed data.
print("Comparing 1,000 random selection sets per region", flush=True)
rng = np.random.default_rng(20261115)
random_rows = []
for replicate in range(1000):
    total = 0
    high = 0
    for region in regions:
        ev = evaluators[region]
        avail = rankings[region][2]
        chosen = rng.choice(avail, size=5, replace=False)
        l, h, _ = ev(chosen)
        total += int(l.sum())
        high += int(h.sum())
    random_rows.append(
        {"replicate": replicate, "priority_losses": total, "high_quotient_misses": high}
    )
pd.DataFrame(random_rows).to_csv(
    OUT / "region_random_baseline.csv.gz", index=False, compression="gzip"
)

# The class-level replacements have unequal group sizes (10 OP versus 42
# other). Compare the ten-OP scenario with random equally sized sets of ten
# non-OP chemicals, with the same set used across all five regions.
rng = np.random.default_rng(2026111502)
nonop = np.flatnonzero(~op)
size_matched = []
for rep in range(1000):
    selected = rng.choice(nonop, size=int(op.sum()), replace=False)
    total = 0
    high = 0
    for region in regions:
        l, h, _ = evaluators[region](selected)
        total += int(l.sum())
        high += int(h.sum())
    size_matched.append(
        {
            "replicate": rep,
            "selected_nonop": len(selected),
            "priority_losses": total,
            "high_quotient_misses": high,
        }
    )
pd.DataFrame(size_matched).to_csv(
    OUT / "region_size_matched_nonop_counterfactual.csv.gz",
    index=False,
    compression="gzip",
)


print("Corrected review sensitivities complete", flush=True)
