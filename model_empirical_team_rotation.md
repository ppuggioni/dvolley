# Model: `empirical_team_rotation`

## What it is
This is the most granular empirical model: team and rotation aware, still purely frequency-based.

## Statistical definition
Target:
- `y = 1` if the server wins the rally,
- `y = 0` otherwise.

From training data, estimate:
- `p_break(team, pos)` = mean(`y`) when team serves in rotation `pos`,
- `p_sideout(team, pos)` = mean(`1 - y`) when team receives in rotation `pos`.

For server `s` in rotation `ps`, receiver `r` in rotation `pr`:

`P(y = 1) = ( p_break(s, ps) + (1 - p_sideout(r, pr)) ) / 2`

Fallbacks:
- missing `p_break(s, ps)` -> `global_mean`,
- missing `p_sideout(r, pr)` -> `1 - global_mean`.

## Parameters
- `break_pos`: dict `(team_id, rotation) -> probability`.
- `sideout_pos`: dict `(team_id, rotation) -> probability`.
- `global_mean`: overall server-win rate.

No optimizer is used; all parameters are direct empirical means.

## Assumptions
- Rotation materially changes serving and receiving performance.
- Historical team-rotation proportions are stable enough for prediction.
- Equal averaging between serving-side and receiving-side components is acceptable.

## Practical interpretation (non-technical)
The model asks:
- "How dangerous is this team when serving from this specific rotation?"
- "How resilient is the opponent when receiving in their current rotation?"

Then it averages those two signals to get breakpoint probability.

## Strengths
- Highest tactical resolution among empirical models.
- Fully interpretable table-by-table.
- Aligns naturally with volleyball coaching workflows by rotation.

## Known limitations
- Sparse cells can be noisy (some team-rotation pairs have few rallies).
- No smoothing/shrinkage across nearby rotations or teams.
- Equal blend rule is fixed, not data-learned.

## Relationship to simulator
In rotation simulation, this model is applied rally-by-rally with current serving team and current rotations, so probabilities update as rotations change after sideouts.
