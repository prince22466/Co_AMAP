# v9 autoresearch experiment log

Keep this tracked summary concise. Store raw benchmark logs, profiler traces,
and outputs under the excluded .autoresearch-work/ folder locally and in
the dedicated SSH scratch directory. Record SHA and path to each artifact.

| ID | Best-base SHA | Hypothesis / code change | GPU tests | 4K tok/s A to B | 16K tok/s A to B | Math-500 | Decision / PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 000 | historical v8 | untouched v9 copy: fresh A/B pending | not run in v9 | 2331.29 historical only | 1251.22 historical only | 0.981 historical only | pending |

Every experiment must record time, actual device, exact Git SHA(s), tested
operator/kernel path, trace evidence, all official commands/CSV identifiers,
MATH-500 evaluation command and output, percent changes, reason for
acceptance or rejection, and PR URL when accepted. Record negative results
to prevent repeating ineffective experiments.
