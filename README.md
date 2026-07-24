# Bought or Built?
### Where a credit-risk model's predictive power actually comes from — Home Credit dataset
*MSDS 696 · Practicum II*

Open up a strong credit-risk model and you find something uncomfortable: a huge share of
its power comes from a handful of **external scores the lender buys** — third-party risk
ratings pulled from a bureau, at a price per applicant. Everything else — the income, the
employment history, the years of painstaking feature engineering over relational data the
lender already owns — fights for the scraps.

So this project asks a blunt, practical question: **how much of the model is bought, and
how much is built?** And the follow-up that actually matters to a lending business: if the
external scores are expensive, **can in-house feature engineering replace them — and if not,
for whom are they irreplaceable?**

## The question I'm actually answering

> On the Home Credit data, three `EXT_SOURCE` features are documented as *normalized scores
> from an external data source* — exactly the kind of third-party rating a lender pays to
> obtain. How much of a default model's predictive power comes from those purchased scores
> versus from features engineered in-house? How much of the gap can good feature engineering
> close without paying? And where do the external scores remain irreplaceable?

No invented economics, no assumed loss rates. Every claim is a measured difference in
predictive performance (OOF AUC) between models trained on different slices of the data.

## The data

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
— roughly 300K loan applications scattered across seven relational tables: the application
itself, plus credit-bureau records, prior loans, monthly balances, and payment histories.
The `EXT_SOURCE_1/2/3` columns are the external (purchased) scores; everything else is data
the lender already holds. A lot of the real work is wrangling the one-to-many sprawl into
one clean row per applicant — and that engineered internal data is exactly what gets pitted
against the purchased scores.

## The approach — a ladder of models

The thesis is answered by training the *same* LightGBM on deliberately different feature
sets and reading the gaps between them:

| Model | Features | The question it answers |
| --- | --- | --- |
| **M0 · Naive baseline** | raw application form, external scores removed | Where do you start with no engineering and without buying the scores? |
| **M1 · External-only** | just `EXT_SOURCE_1/2/3` + their interactions | How much do three purchased columns buy you on their own? |
| **M2 · Engineered internal** | all in-house engineered features, external removed | How far does feature engineering get you *without paying*? |
| **M3 · Full** | everything — internal + external | The ceiling (the tuned headline model, ~0.79 OOF AUC). |

Read as gaps, this is the whole story:
- **M1 − M0** — how much a lender gains by simply *buying* the scores instead of engineering.
- **M2 − M0** — how much *building* (feature engineering) recovers on its own.
- **M3 − M2** — the external scores' **marginal** value once you've done the engineering.
- **M2 vs M1** — the headline face-off: **can built beat bought?**

### The twist — thin files

The averages hide the interesting part. The external scores should matter *most* for
**thin-file** applicants — people with little internal credit history, where the lender's own
data is sparse — and *least* for thick-file applicants it already knows well. So the M3 − M2
gap gets re-measured within thin-file vs. thick-file groups. The expected finding: **feature
engineering can substitute for the purchased scores for well-documented applicants, but the
external data stays irreplaceable exactly where the lender is otherwise blind** — which ties
straight back to the project's earlier missingness-as-signal finding (a blank isn't nothing;
it marks the customer whose purchased score you can least afford to skip).

### Reverse-engineering the scores

The ablation ladder measures *how much* the external scores are worth. The deeper question is
*what they are*. So the model gets flipped around: instead of predicting default, predict
`EXT_SOURCE` itself from the internal features. If a score can be reconstructed accurately from
data the lender already owns, it's partly redundant — you're paying for something you could
rebuild. If it can't, it carries genuinely novel information from outside the lender's walls.
SHAP on that reconstruction model reveals what each purchased score is *made of* — is it really
just age, income, and employment in a trench coat, or something the internal data can't reach?
This reverse-engineering, plus the standard interpretation and segment work, is where the bulk
of the project's remaining effort goes.

## On the leaderboard score

The competition closed in 2018, so I can only make late submissions — scored against the
private leaderboard but unranked. Wherever the full model lands I'll describe as "equivalent
to ~Nth place," a clean outside check that the modeling foundation holds up. The bought-vs-built
analysis built on top is the part that actually matters, and the leaderboard doesn't measure it.

## Repository layout

```
notebooks/    numbered in run order:
                01      auto-EDA profile of the raw Home Credit tables
                02–08   feature engineering, one notebook per table
                09      feature selection (coarse gain drop → randomized backward stability)
                10      models — tuned LightGBM (TUNE flag → out-of-fold → seed-bag)
                11      bought vs. built — the M0→M3 ladder + the thin-file split
                12      reverse-engineering — reconstruct EXT_SOURCE from internal data
                13      interpretation — SHAP feature effects (external dominance)
                14      segments — subgroup performance (age, income, gender)
src/          reusable code — aggregation, higher moments, feature selection, master assembly
data/         Home Credit tables + local context (reports, notes) — not committed
eda/          generated auto-EDA reports (HTML + PDF) — not committed, rebuilt from notebook 01
```

The seven per-table feature notebooks (02–08) assemble into a master feature set; null-importance
selection trims it to ~354 features and a tuned LightGBM reaches ~0.79 OOF AUC. The project is
scoped strictly to modeling: EDA → feature engineering and selection → model optimization → a
post-model explanation centered on *where the predictive power comes from* (the bought-vs-built
ladder, SHAP, and segment analysis).

## Getting started

```bash
# 1. create + activate the conda environment (Python 3.11 + the data stack)
conda env create -f environment.yml
conda activate credit-signal

# 2. register the Jupyter kernel (so it's selectable in Jupyter / VS Code)
python -m ipykernel install --user --name credit-signal \
  --display-name "Python (credit-signal)"
```

Download the **[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)**
CSVs into `data/raw/` (the data is not committed). Then open a notebook under `notebooks/`
and select the **Python (credit-signal)** kernel.

*(A pip `requirements.txt` is also provided as an alternative to conda.)*

## References & credits

- **Dataset** — [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk),
  Home Credit Group, via Kaggle.
- **Feature-engineering reference** — [minerva-ml/open-solution-home-credit](https://github.com/minerva-ml/open-solution-home-credit)
  (open-source Home Credit solution). Several hand-crafted application ratios and the
  per-table aggregation / recency-window / trend / distribution features (bureau,
  previous_application, POS, installments, credit-card) are **adapted and extended** from
  this solution; the specific borrowings are noted inline in the relevant `notebooks/`. All
  shared helpers (`src/aggregate.py`, `src/moments.py`, `src/select.py`) and the bought-vs-built
  analysis are original.
