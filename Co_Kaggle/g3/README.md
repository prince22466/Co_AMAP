# Competition Participation Notes

## Summary

| Item | Result |
| --- | --- |
| Participation period | June 23-July 15, 2026 |
| Active participation | 2.5 weeks |
| Tasks solved | 378 / 400 |
| Final standing | Not recorded |
| Notable event | GPT-5.6 became available on July 10, 2026 |

## How AI Was Used

Codex, ChatGPT, ChatGPT-Chat(5.6), and ChatGPT-Work(5.6) throughout the competition.

- ALL submission notebooks were written primarily with Codex 5.5 or ChatGPT 5.5.
- ChatGPT served as the fallback when the Codex quota was exhausted. Its code generation was slower, but effective.
- All publicly available online files are found ChatGPT-Chat 5.6.
- ChatGPT-Chat and ChatGPT-Work were used for subtype task-level optimization.
- ChatGPT-Work depends on Codex availability. When the Codex quota was exhausted, optimization continued with ChatGPT-Chat.

## Optimization done by ChatGPT-Chat and ChatGPT-Work

| Task subtype | Before | After | Gain | Reached 70% target | Time spent | Tool |
| --- | ---: | ---: | ---: | :---: | --- | --- |
| `03_same_shape_exact_local_new_colors` | 166.95 | 213.08 | +46.13 | Yes | Half a day | ChatGPT-Work |
| `05_fill_additive_marking_local_3x3` | 115.00 | 140.15 | +25.15 | Yes | Several hours | ChatGPT-Work |
| `05_fill_additive_marking_nonlocal_1color` | 524.00 | 666.07 | +142.07 | No | 2-3 days | ChatGPT-Chat |
| `09_pattern-continuation-nonlocal_additive` | 653.00 | 887.77 | +234.77 | No | 1-2 days | ChatGPT-Chat |
| `09_pattern-continuation-nonlocal_recolor` | 170.00 | 237.81 | +67.81 | No | 1 day | ChatGPT-Work |

## Takeaway

Time and quota constraints limited the amount of optimization that could be completed.  
Even so, every recorded optimization improved its score, and two task subtypes reached the 70% target.  
I believe '05_fill_additive_marking_nonlocal_1color' and '09_pattern-continuation-nonlocal_additive' can reach targets given a bit more time.  
The results show clear potential for a longer-running, more autonomous optimization workflow.  

## Next Experiment

Build an autonomous research agent with Google's ADK and evaluate whether it can optimize tasks with minimal supervision.

Proposed experiment:

1. Select approximately 20 low-scoring tasks.
2. Give the agent access to their task data, current solvers, scores, and evaluation workflow.
3. Let it propose, implement, and test iterative solver improvements.
4. Measure score gains, time spent, compute usage, and how often human intervention is required.
