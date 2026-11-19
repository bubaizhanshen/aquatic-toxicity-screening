# Aquatic toxicity screening

Analysis code for **Structure-Related Errors in Aquatic Toxicity Predictions Can
Obscure Priority Chemicals in Surface Water Screening**. The workflow fits ten
models, checks their structural applicability domains, and compares screening
decisions before and after selected toxicity predictions are replaced with
existing measurements.

## Environment

The pins in `requirements.txt` follow the package versions recorded for the
Python 3.11 conventional refits. This environment also runs the screening analyses.

```bash
python -m pip install -r requirements.txt
```

The neural models also use PyTorch and PyTorch Geometric. The CUDA refits used
Python 3.8.20, RDKit 2023.9.6, NumPy 1.24.3, pandas 2.0.3, SciPy 1.10.1,
PyTorch 2.4.1, and PyTorch Geometric 2.5.3. Use a separate environment for that
configuration. Fitted predictions can vary with the numerical software and
hardware.

## Data sources and inputs

The repository contains code. The calculations use curated chemical, toxicity,
and monitoring tables. The Methods and Supporting Information describe source
selection, unit corrections, identity matching, and event definitions.

| Records | Source |
| --- | --- |
| Broad fish, water-flea, and algal toxicity | [Lunghini et al. data release](https://doi.org/10.5281/zenodo.3708082) |
| Single-species toxicity | [Svedberg et al. data article](https://doi.org/10.1016/j.dib.2023.109719), [EnviroTox](https://envirotoxdatabase.org/), and [EPA ECOTOX](https://cfpub.epa.gov/ecotox/) |
| External toxicity and prediction comparisons | [Chen et al.](https://doi.org/10.1007/s10646-024-02751-1), [Jia et al.](https://doi.org/10.1016/j.chemosphere.2021.132980), [curated fish data](https://doi.org/10.1016/j.comtox.2026.100436), and [TRIDENT](https://doi.org/10.1126/sciadv.adk6669) |
| U.S. regional samples, 2013–2017 | [USGS data release](https://doi.org/10.5066/P9D2BDBY) |
| Historical and recent U.S. detections | [Water Quality Portal](https://www.waterqualitydata.us/) |
| California detections and benchmarks | [CEDEN chemistry records](https://data.ca.gov/@california-state-water-resources-control-board/surface-water-chemistry-results) and [EPA aquatic life benchmarks](https://www.epa.gov/pesticide-science-and-assessing-pesticide-risks/aquatic-life-benchmarks-and-ecological-risk) |
| Dutch and Saxony monitoring | [Waterkwaliteitsportaal](https://www.waterkwaliteitsportaal.nl/download-data) and [Saxony PANGAEA release](https://doi.org/10.1594/PANGAEA.969111) |

Use the first InChIKey block as `conn`, a structure string as `smiles`, and
`p_value = -log10(effect concentration in mmol/L)` for measured toxicity.
Prediction columns are `pred_<model>_p`, where the model names are
`random_forest`, `lightgbm`, `xgboost`, `svr`, `extra_trees`, `ridge`,
`catboost`, `mlp`, `gine`, and `gat`. Use the molecular weight in
`mw` to convert mass concentrations to molar concentrations.

## Model fitting and error analysis

1. Run `src/prepare_labels.py --records RECORDS --source broad --output LABELS`
   for extracted broad-endpoint records with `endpoint`, `conn`, `smiles`,
   `p_value`, `mg_l`, and `p_minus_converted_p`. Use `--source study`
   with the unit-corrected single-species records and their `database` and
   `source_reference` columns. Resolve test-material identities and apply the
   source exclusions before aggregation.
2. Run `src/fit_models.py --train LABELS --output PREDICTIONS` separately for
   each cohort. Use `--fold 0` through `--fold 4` for separate folds, or supply
   `--query QUERIES` with `conn` and `smiles` for a joint or external holdout.
   For the Dutch refits use `--seed-offset 7 --epoch-limits 300 600`. Use
   `--component classical` and `--component neural --device cuda` to run the
   components separately. Join predictions by `conn` before error analysis,
   retaining one copy of each identity and observed-toxicity column. The
   `--legacy-descriptors` option retains the single-precision descriptor
   scaling used for the initial expanded conventional fits.
3. Run `src/structural_errors.py --predictions COHORT_DIR --output OUT`.
   Place the complete out-of-fold tables in that directory as
   `broad_fish.csv.gz`, `broad_water_flea.csv.gz`, `broad_algae.csv.gz`,
   `fathead_minnow.csv.gz`, and `daphnia_magna.csv.gz`. The script summarizes
   structural-group error rates with paired chemical resampling, similarity
   cutoffs, model agreement, and model-level error counts. Run
   `src/structure_pairs.py --predictions DAPHNIA_PREDICTIONS --output OUT`
   for the same-fold comparison and the dimethoate–omethoate example.

4. Run `src/external_errors.py water-flea --predictions TABLE --output RESULT`
   for the independent water-flea set. Use the reference column
   `reference_p_mmol_L`, model columns `pred_<model>_p` and `error_<model>_p`
   (predicted minus reference), and the `OP_feature`, `inside_AD`, `mean5`,
   `nearest`, and `absent_other_project_cohorts` flags or similarities.
   For TRIDENT, use the `trident` comparison with repetition-level D. magna
   48 h intoxication records joined to the reconstructed domain. Required
   fields are `conn`, `seed`, `SMILES_Canonical_RDKit`, `single_component_organic`,
   `measured_log_mg_l`, `predicted_log_mg_l`, `mw`, `nearest`,
   `organophosphorus`, `project_overlap`, `in_coverage`, and `conn_in_training`.

## Monitoring analyses

Use identity-resolved, deduplicated detections. Each script has a `--help` entry
point and writes its results to the specified output location.

| Script | Inputs and comparison |
| --- | --- |
| `screen_us.py` | `--monitoring` uses `region`, `SITE_NO`, `YYYYMMDD`, `TIME`, `conn`, `concentration_ng_l`, `mw`, `estimated`, `ad_inside_5nn`, `chemical_names`, and `fragment_organophosphorus`. Supply the 52-chemical joint `--predictions` and a corrected `--reference` table. Replays fixed allocations, sequential selection, 1,000 random lists, and event sensitivities. `us_five_test_outcomes.csv` compares the two fixed lists and sequential selection at five tests per region. `us_sequential_tests.csv` records the five choices in order. |
| `screen_national.py` | `--historical` and `--recent` use `conn`, `concentration_ug_l`, and `mw`. Supply the 349-chemical `--train` set, the 52-chemical `--query` with reference toxicity and model predictions, and a `--selections` JSON object mapping rules to lists of connectivities. Replays threshold and testing-rule comparisons. |
| `screen_california.py` | `--monitoring` uses CEDEN `StationCode`, `DW_AnalyteName`, `cas_key`, `conn`, `concentration_ug_l`, `mw`, `species`, `cohort`, `stereo_four_sensitivity_exclusion`, `has_reference`, and `epa_exceeded`. Supply corrected `--predictions` and `--source-ranges` with `conn`, `p_min`, `p_max`, and `n_studies`. |
| `screen_temporal_fixed.py` and `screen_temporal_sequential.py` | `--monitoring` uses `area`, `period` (`selection` or `evaluation`), `site`, `date`, `time`, `sampling_end_date`, `conn`, `concentration_ug_l`, and `mw`. Supply joint `--classical` and `--neural` predictions plus `--reference` with `conn` and `p_value`. For sequential testing, first run the fixed script on the reference-available panel and pass its `04_frozen_selections.json` as `--fixed-lists`. Use the full matched observations for the complete-event analysis. Use `--expected-areas 1` for Saxony and `--country DE` for its sequential analysis. |

The screening replay uses the same prepared-record definitions as the paper.
The raw public records require the curation and matching described in the
Supporting Information before they can be passed to these scripts.
