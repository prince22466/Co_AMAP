# Evaluation and Leaderboard Scoring

**Official source:** [Kaggle competition — Evaluation](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/overview/evaluation)  
**Cross-check:** [Official starter kit — Scoring](https://github.com/smart-buildings-2026/starter-kit#scoring)

## 1. Objective

The policy must balance **occupant comfort**, **energy cost**, and **carbon emissions** across previously unseen building–weather scenarios.

**Leaderboard metric:** Mean absolute error against a target of **zero**, computed from each scenario's mean per-step reward. **Lower is better.**

## 2. Per-step reward

Every **5 minutes**, the simulator produces:

\[
r_t = 0.6\,p_t - 0.2\,c_t - 0.2\,e_t
\]

where:

| Term | Meaning | Contribution to reward |
| --- | --- | --- |
| \`p_t\` | Occupant comfort (a nonpositive regret term) | \`0.6 p_t\` |
| \`c_t\` | Energy cost, including electricity and gas (nonnegative) | \`-0.2 c_t\` |
| \`e_t\` | Carbon emissions (nonnegative) | \`-0.2 e_t\` |

### Comfort penalty

- An occupied zone within its comfort-temperature band incurs **zero comfort regret**.
- Up to **5 K outside** the band, comfort degrades **linearly**.
- Beyond **5 K**, the penalty grows **quadratically and without bound**.
- Zone comfort is weighted by each zone's **occupancy**, **not its floor area**.
- The scoring weights are fixed and identical for all submissions.

Consequently, letting even one occupied zone become extremely uncomfortable can dominate an episode's reward.

### Why the reward is never positive

Comfort is a penalty (\`p_t <= 0\`), while energy cost and emissions are nonnegative (\`c_t >= 0\`, \`e_t >= 0\`). Therefore:

\[
r_t \le 0.
\]

A reward of zero would require perfect occupied-zone comfort **and** zero energy cost **and** zero carbon emissions. The competition treats this as the ideal, but not an attainable practical controller result.

## 3. Episode return

Each scenario lasts **30 days**, or:

\[
30 \times 24 \times 12 = 8{,}640 \text{ steps}.
\]

All steps are scored, **including the first**; there is no unscored warm-up.

For scenario \`s\`, the episode return is:

\[
R_s = \sum_{t=1}^{8640} r_{s,t},
\qquad
\bar r_s = \frac{R_s}{8640}.
\]

Here, \(\bar r_s\) is the **mean reward per step**, not the episode total.

## 4. Leaderboard score

Let \`S\` be the evaluated set of building–weather scenarios. The score is:

\[
\operatorname{score}
= \frac{1}{|S|}\sum_{s\in S}\left|\bar r_s-0\right|
= -\frac{1}{|S|}\sum_{s\in S}\bar r_s.
\]

The second equality follows from \(\bar r_s \le 0\).

**Interpretation:** The score is the average *magnitude of per-step loss*. **Minimizing the leaderboard score** is equivalent to maximizing the average scenario return.

## 5. Final evaluation

The official hidden evaluation uses:

| Item | Value |
| --- | --- |
| Unseen buildings | **6** |
| Unseen weather stations | **5** |
| Total scenarios | **30** |
| Episode length | **8,640 steps (30 days)** |
| Score aggregation | **Equal-weight mean across all 30 scenarios** |
| Direction | **Lower is better** |

The simulator is deterministic: the same policy and scenario produce the same return. Therefore, each building–weather combination is run **once**, without averaging repeated rollouts.

Before announcing results, organizers re-run the top submissions end to end.

## 6. Local scoring versus hidden scoring

The [official starter kit](https://github.com/smart-buildings-2026/starter-kit) supplies **30 public training scenarios**, reference baselines, and a local scorer. These are for development and comparison; **the hidden evaluation determines the leaderboard score**.

The starter kit's default example controller is deliberately weak. A valid action dictionary and a successful local rollout **do not imply a competitive policy**.

## Source notes

The reward coefficients, nonpositive-reward property, complete leaderboard equation, episode length, and 30-scenario weighting above have been checked against the official Kaggle Evaluation section. This file restores the two mathematical equations that were omitted from the original Markdown copy.
