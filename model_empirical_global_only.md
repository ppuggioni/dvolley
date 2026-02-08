# Model: `empirical_global_only`

## What it is
This is the simplest empirical baseline: one single probability for all rallies.

## Statistical definition
Let:
- `y = 1` if the serving team wins the rally,
- `y = 0` otherwise.

The fitted parameter is:
- `p_global = mean(y)` on the training data.

Prediction for every future rally is exactly:
- `P(y = 1) = p_global`.

No team, rotation, or match context is used.

## Parameters
Single scalar:
- `global_mean` (the historical breakpoint frequency of servers).

## Assumptions
- Rally-level breakpoint probability is constant across teams, rotations, and contexts.
- Training sample average is representative of deployment data.

## Why this model exists
- It is a calibration anchor.
- It provides a strict lower-complexity benchmark.
- If a more complex model does not beat this on log loss or Brier score, the extra complexity is not justified.

## Practical interpretation (non-technical)
This model says: "Ignore who serves, who receives, and rotation. Use one league-wide historical rate."

It is useful for:
- sanity checks,
- monitoring drift,
- quick baselines before comparing richer models.

## Known limitations
- No tactical sensitivity.
- Cannot capture team identity or rotation effects.
- Can be badly biased for specific team matchups even if globally calibrated.
