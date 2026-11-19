# Aquatic toxicity screening

Analysis code for *Structure-Related Errors in Aquatic Toxicity Predictions Can Obscure Priority Chemicals in Surface Water Screening*.

## Data

Toxicity records are available from [Lunghini et al.](https://doi.org/10.5281/zenodo.3708082), [Svedberg et al.](https://doi.org/10.1016/j.dib.2023.109719), [EnviroTox](https://envirotoxdatabase.org/), and [ECOTOX](https://cfpub.epa.gov/ecotox/).

Monitoring records are available from [USGS](https://doi.org/10.5066/P9D2BDBY), [Water Quality Portal](https://www.waterqualitydata.us/), [CEDEN](https://data.ca.gov/@california-state-water-resources-control-board/surface-water-chemistry-results), [Waterkwaliteitsportaal](https://www.waterkwaliteitsportaal.nl/download-data), and [PANGAEA](https://doi.org/10.1594/PANGAEA.969111). Prepare the input tables as described in the manuscript and Supporting Information.

## Run the code

```bash
python -m pip install -r requirements.txt
python src/screen_us.py --help
```

The [analysis scripts](src/) cover model fitting, error analysis, and fixed and sequential test selection. See the [full commands and input formats](https://github.com/bubaizhanshen/aquatic-toxicity-screening/blob/eb6f953456bab7971bc687bb5fb132cdefb6b78d/README.md).
