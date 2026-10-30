https://www.kaggle.com/competitions/neurips-2026-smart-buildings/overview

# Evaluation

Submissions are scored by mean absolute error against a target of zero across held-out (building, weather) scenarios. Lower is better.

Reward
At each 5-minute step, the environment gives a reward:


where (p_t) is occupant comfort, (c_t) is energy cost (electricity and gas), and (e_t) is carbon emissions. Comfort drops linearly when zone temperatures are up to 5 K outside the comfort band, and quadratically and without bound beyond that, so one badly neglected zone can dominate an episode. Comfort is summed over zones, weighted by each zone's occupancy rather than its floor area. The weights are fixed and the same for every submission.

Each scenario runs for 8,640 steps, which is 30 days. Every step is scored, including the first. The episode return (R) is the sum of rewards over all 8,640 steps.

The reward is never positive
All three terms are penalties. Comfort (p_t) is a regret: it is 0 when every occupied zone sits inside its comfort band, and negative otherwise, without bound. Energy cost and carbon emissions are non-negative and enter with a minus sign, because the building cannot export energy.

So (r_t \le 0) at every step. The only way to score 0 is perfect comfort delivered at no energy and no carbon cost, which no controller can reach. Treat 0 as the ideal and your score as the distance from it.

Score
For each scenario, take your mean per-step reward, which is the episode return divided by 8,640. The leaderboard metric is the mean absolute error of those values against a target of zero:


The second form holds because the reward is never positive, so the metric is simply the average size of your per-step loss. Lower is better. Minimizing it is the same as maximizing the return on every scenario.

Final Score
Your policy is evaluated on 6 unseen buildings × 5 unseen weather stations = 30 scenarios. Your score is the equal-weight mean over all 30.

The simulator is deterministic. The same policy on the same scenario reproduces the same return bit for bit, so each (building, weather) pair is run exactly once and no averaging over repeated rollouts is needed.

Before results are announced, the organizers re-run the top submissions end to end.
