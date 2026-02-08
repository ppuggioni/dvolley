# Model: `empirical_team`

## What it is
This is a pure empirical, team-level probability model with no rotation detail.

## Statistical definition
Target:
- `y = 1` if the server wins the rally,
- `y = 0` otherwise.

From training data, compute:
- `p_break(team)` = mean(`y`) when that team serves,
- `p_sideout(team)` = mean(`1 - y`) when that team receives.

For a new rally with server `s` and receiver `r`, prediction is:

`P(y = 1) = ( p_break(s) + (1 - p_sideout(r)) ) / 2`

Fallbacks:
- unknown `p_break(s)` -> `global_mean`,
- unknown `p_sideout(r)` -> `1 - global_mean`.

## Parameters
- `break_team`: dict `team_id -> p_break`.
- `sideout_team`: dict `team_id -> p_sideout`.
- `global_mean`: overall server-win rate.

No optimization, no shrinkage, no Bayesian layer. Parameters are direct sample proportions.

## Assumptions
- Team-level serving and receiving strength are stable enough in sample.
- Rotation effects are either negligible or intentionally ignored.
- Equal blending (`50%`/`50%`) between serving strength and receiving weakness is acceptable.

## Practical interpretation (non-technical)
Each rally prediction balances:
1. how often the serving team scores on serve, and
2. how often the receiving team usually avoids getting broken.

It is easy to explain and trace back to observed frequencies.

## Strengths
- Transparent and robust to implementation mistakes.
- Fast and deterministic.
- Good intermediate baseline between global and rotation-aware models.

## Known limitations
- Ignores rotation context entirely.
- Sensitive to small sample sizes for low-volume teams.
- Equal weighting rule is heuristic, not learned from data.
