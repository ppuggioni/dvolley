# Model: `logistic_rotation_alpha_x`

## What it is
This is a regularized logistic regression for rally outcome, where the target is:
- `y = 1` if the serving team wins the rally (breakpoint point),
- `y = 0` otherwise.

It models breakpoint probability from team identity and rotation of both server and receiver.

## Statistical specification
For rally `i`, the model is:

`logit(P(y_i = 1)) = b0 + bp_team(server_i) + bp_pos(server_i, rot_server_i) - so_team(receiver_i) - so_pos(receiver_i, rot_receiver_i)`

Where:
- `b0` is global breakpoint baseline.
- `bp_*` terms increase server-win probability.
- `so_*` terms decrease server-win probability (higher sideout quality for receiver).

The sign convention is explicit: breakpoint effects are additive, sideout effects are subtractive.

## Parameters and constraints
The parameterization is identifiable through sum-to-zero constraints:
- team breakpoint effects sum to zero across teams,
- team sideout effects sum to zero across teams,
- for each team, the 6 rotation breakpoint effects sum to zero,
- for each team, the 6 rotation sideout effects sum to zero.

In code this is implemented by constrained coding (T-1 team columns, 5 rotation columns per team, with the last level implied).

## Estimation
- Estimator: `sklearn.linear_model.LogisticRegression` (`lbfgs`, intercept on).
- Penalization: L2, controlled by `alpha`.
- Mapping used: `C = 1 / (alpha * n_samples)`.
- Time weighting: exponential half-life weighting by match date:
  `w_i = 0.5^(age_days_i / half_life_days)`.

The model family is the same for all app options:
- `logistic_rotation_alpha_0.1`
- `logistic_rotation_alpha_0.05`
- `logistic_rotation_alpha_0.01`
- `logistic_rotation_alpha_0.005`
- `logistic_rotation_alpha_0.001`

Only regularization strength changes.

## Assumptions
- Rally outcomes are conditionally independent given encoded team/rotation context.
- Effects are additive on log-odds scale.
- No explicit home-court term (by design).
- Rotation effects are team-specific and stationary over the fitted period.

## Practical interpretation (non-technical)
Think of this as a calibrated baseline plus:
1. how strong a team is when serving,
2. how strong a team is when receiving,
3. how each of those changes by rotation.

Higher `breakpoint_*` means better at scoring on serve.  
Higher `sideout_*` means better at defending sideout (harder for server to score).

## Known limitations
- It does not include player-level or touch-sequence covariates.
- Sparse team/rotation cells are stabilized only through global L2 shrinkage.
- New unseen teams default toward baseline behavior (no learned team effect).
