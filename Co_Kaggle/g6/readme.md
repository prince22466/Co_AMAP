1st submission on Oct . game ends on .


constraints: 
L4 GPU has only half T4 GPU usage(aka 15h perweek vs 30 perweek). 
2 Large Gemma4 consume lots of times(espesically GPU time) when testing.
Thus using small ones in testing is more feasible. Some GPU time, small model can do 3times of tasks as large ones.

Gemma 4 31B cannot sit in T4 GPUs(too small), 12B has data format or type problem when using T4 which needs parameter input for vllm which may or may not be able to adjust for real submission.
