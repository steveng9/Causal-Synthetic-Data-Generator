# Small causal graphs

46 graphs of 2 to 6 nodes (`graphs.json`), each sampled with `csdg.py` at five cut strengths (0, 0.25, 0.5, 0.75, 1) under each of the three cuts, and drawn as one PDF (`small_graphs.pdf`).

```sh
python run.py     # about 2.5 minutes; writes results.json
python plot.py    # writes small_graphs.pdf
```

**Setup.** The world is the graph with every edge intact. The release is the same graph sampled with the rule's edges cut on that share of rows (`cut_strength`). Cuts are computed on the visible columns only, so a hidden U stays in both world and release. Every edge explains a 0.25 share of its child's latent variance, lowered so that no child's parents add to more than 0.9. S and Y are binary, 50/50; every other column is continuous and normal. 20,000 rows, 5 seeds; the world table for seed k uses seed 1000 + k.

**Metrics,** all scored on world data. A recipient fits logistic regression for Y on every released column, S included.

- Fairness gap: difference in mean predicted P(Y=1) between the S groups, in percentage points.
- S inference: Brier skill of an attacker who fits logistic regression on the release to predict S from the other released columns. 0 means no better than the base rate.
- Utility: Brier skill of the recipient's model for Y.

Brier skill is 1 − Brier / Brier of always predicting the base rate. Ranking metrics such as AUC do not change under a partial cut when a model has one feature, so they hide the trend.
