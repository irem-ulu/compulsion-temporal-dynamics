# compulsion-temporal-dynamics

[![CI](https://github.com/irem-ulu/compulsion-temporal-dynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/irem-ulu/compulsion-temporal-dynamics/actions/workflows/ci.yml)

Practice project. Small synthetic EMA dataset (stress, outside-time,
compulsions) that I'd previously analysed with plain OLS — redone here with
mixed-effects models because the data is nested within people and OLS treats
it as if it weren't. Mostly an excuse to put together a full-shape Python
project (package, CLI, tests, CI, streamlit thing, auto-generated report)
around something I actually wanted to learn.

The most fun bit is `ctd compare`. The original OLS reports a lag-1
"previous stress predicts compulsions" effect at p < .001. The within-person
mixed model on the same rows gives p ≈ .60. The pooled slope was almost
entirely between-person variance — anxious people just compulse more, so
when you pool the rows you pick up the cross-section. Standard EMA gotcha
but nice to see it land cleanly on a dataset small enough to read.

## Running

```bash
git clone …
cd compulsion-temporal-dynamics
make setup       # venv + install
make run         # full pipeline -> results/
make compare     # OLS vs Mixed side-by-side
make dashboard   # streamlit playground
make test
```

Without make:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"
ctd run
```

`results/REPORT.md` ties things together after `ctd run`. Plain text,
CSVs and PNGs sit next to it.

## Layout

`src/ctd/` is the package. Inside:

- `data/` — both data generators (the original prototype and a richer one
  with person-level random effects and AR(1) stress)
- `preprocess.py` — cleaning, time features, within/between split
- `analysis/` — models, bootstrap, sensitivity, power
- `viz/` — plots + the DAG drawing
- `legacy.py` — OLS-vs-Mixed comparison
- `pipeline.py`, `cli.py` — the pipeline and `ctd` CLI

`dashboard/app.py` is the streamlit playground. `tests/` is pytest.
CI runs lint + tests + the pipeline on push.

## CLI

```
ctd simulate    write a synthetic dataset
ctd describe    pooled corr + ICC for an existing CSV
ctd run         full pipeline
ctd compare     OLS vs mixed model on the same data
ctd power       power grid only (slow)
ctd report      rebuild REPORT.md from results/
```

`ctd <command> --help` for options.

## Notes

statsmodels.MixedLM is REML-only and won't do Poisson, which is really what
you'd want for the compulsion counts. For real data I'd reach for lme4 or
glmmTMB in R. I stayed in Python here to keep the whole thing in one
language and because the continuous outcomes carry most of the story.

Power numbers are calibrated against my own simulator, so read them as a
check on the inferential machinery — not something you'd put in a grant.

Cluster bootstrap with small N has known coverage issues. Wild cluster
bootstrap would be a strict improvement. Didn't write it.

The original had some notebooks. I didn't port them — the saved figures
and the generated report cover the same ground.

## License

MIT.
