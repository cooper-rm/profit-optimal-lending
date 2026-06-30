# Lending for Profit, Not Accuracy
### A quant take on credit-risk decisioning — Home Credit dataset
*MSDS 696 · Practicum II*

Most credit models answer the wrong question. They predict *who will default* and
stop there, as if a 0.5 cutoff came down from the heavens. But a lender doesn't get
paid for sorting people into "good" and "bad" — it gets paid for making loans that
turn a profit. A small, high-rate loan to a shaky borrower can be worth more than a
big loan to a safe one. The standard pipeline can't see that. This project can.

I start by building a probability-of-default model good enough to hold its own on
the Home Credit leaderboard. Then I take it one step past where most work stops:
I turn those probabilities into actual dollar decisions and test whether optimizing
for *profit* beats optimizing for *accuracy* — and by how much.

## The question I'm actually answering

> For a consumer lender's credit-risk desk: which applicants are profitable to
> approve for a loan once you weigh each one's default probability against their loan size and
> the lopsided cost of a default — and does a profit-driven lending policy beat an
> accuracy-driven one on risk-adjusted return?

The stakeholder is the risk desk. The decision is approve, decline, or size the
loan. Everything here is built to answer that, not to climb a ranking.

## The data

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
— roughly 300K loan applications scattered across seven relational tables: the
application itself, plus credit-bureau records, prior loans, monthly balances, and
payment histories. A lot of the real work is wrangling that one-to-many sprawl into
one clean row per applicant.

## How it works

Two layers, joined by a single idea.

**The model.** A default-risk model built on features engineered from all seven
tables and judged on AUC — the same metric the competition used. That lets me
benchmark my core modeling against thousands of other submissions on a held-out
test set I never get to see, a real yardstick for whether the foundation holds up.
Rather than commit to one algorithm upfront, I test a range against this benchmark
on the same features and let the results pick the winner.

**The decision.** A strong AUC means the model *ranks* borrowers well — but ranking
isn't deciding. To decide, the probabilities have to be trustworthy: calibrated so
that "20% risk" really does mean one in five. Once they are, I can put a price on
each loan:

```
expected profit  =  (1 − PD) × gain if they repay   −   PD × loss if they default
```

The policy falls right out of that line: approve every loan with positive expected
profit. And the cutoff that maximizes profit is *not* the one that maximizes
accuracy — the cost of a default is too lopsided for that. The payoff of the whole
project is the gap between those two policies, measured in portfolio P&L and
risk-adjusted return.

What makes it worth doing: the two policies genuinely disagree about who to lend to.
That disagreement *is* the finding.

## On the leaderboard score

The competition closed in 2018, so I can only make late submissions. They score
against the private leaderboard but don't officially rank, so wherever mine land
I'll describe as "equivalent to ~Nth place" rather than a real finish. It's a clean
outside check on the model and nothing more — the decision layer built on top is
the part that actually matters, and the leaderboard doesn't measure it.

---

*Methodology decisions, rationale, and a couple of extensions I'm exploring
(risk-based pricing, macro stress-testing) are in [`IDEAS.md`](IDEAS.md).*
