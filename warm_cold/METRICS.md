# V-B2 warm/cold metrics

## Warm/cold threshold

The default threshold is `3`, and it is unchanged from the existing scorer.

- Paper: `docs/CoLLM_paper.pdf`, p. 2336, Sec. V-B2: "the warm subset comprises
  interactions between users and items that have appeared at least three times
  in the training dataset, while the cold subset includes the remaining
  interactions."
- Existing code default: `eval_warm_cold_vb2.py:240`.
- Operational rule used here:

```text
warm = (count_train(uid) >= THRESHOLD) AND (count_train(iid) >= THRESHOLD)
cold = exact complement
```

The paper wording implies both the user and the item must meet the threshold.
The threshold is a CLI parameter with default `3`; `results.json` records both
the value used and this source. Changing it prints a warning but does not
change the default.

## AUC

`AUC = sklearn.metrics.roc_auc_score(label, logit)` over the subset.

## Primary UAUC (reported as `UAUC`)

For each user with at least two interactions in the subset:

1. If the user has only one class in the subset, skip the user (AUC is undefined).
2. Otherwise compute `roc_auc_score(label_user, logit_user)`.
3. Average the finite per-user AUC values.

This is the primary thesis table value. Users with a single interaction are
skipped; they cannot have both classes. If no user has both classes, the value
is `NaN` and a warning is printed.

For every subset, `results.csv` and `results.json` report:

- unique users,
- users computed (finite per-user AUC),
- users skipped because they have a single class,
- users with a single interaction.

## Secondary UAUC (`UAUC (official)`)

The repo's `uAUC_me` implementation now lives in `uauc.py` and is imported both
by `minigpt4/tasks/rec_base_task.py` and by `warm_cold/metrics.py`. Both
evaluation paths therefore use the same per-user loop and aggregation:

- per-user AUCs are computed exactly as before;
- single-interaction users are skipped;
- NaN per-user AUCs (including single-class users) are ignored with
  `np.nanmean`;
- if no defined per-user AUC exists, the result is `NaN` and no avoidable
  `RuntimeWarning` is emitted.

The primary and secondary UAUC values therefore use the same eligible-user
population and agree when per-user AUCs are defined. The per-user counts in
`results.csv` still report how many users were included or skipped.

The `results.csv` files under `raw_logs/warm_cold_logs/` were generated before
this fix and retain the historical `NaN` values and column label.
