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

## Secondary UAUC (`UAUC (official, NaN-propagating)`)

The repo's original `uAUC_me` (`minigpt4/tasks/rec_base_task.py:25-67`) is
vendored in `warm_cold/metrics.py` and reported separately. Its aggregate is
`np.mean(per_user_aucs)`, so one `NaN` per-user value poisons the subset value.
This explains the historical `NaN` UAUC despite many computable users.

The vendored function preserves the official grouping, skip rules, and
NaN-propagating mean. A parity test compares it with the official function
when the official import is available (for example on Kaggle).
