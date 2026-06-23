Evaluation
For any of the 400 tasks in the ARC-AGI public training v1 benchmark suite, your team will earn a score of max(1, 25 - ln(cost)) for a functionally correct network whose cost is the sum of the following:

The total number of parameters in the network
The total memory footprint of the network (in bytes)
Functional correctness will be determined by validating the network against the original ARC-AGI benchmarks and a small private benchmark suite (so as to prevent teams from overfitting their solutions). To be eligible for points, your network must produce correct results across all of these tests.

Submission File
You must submit a file named submission.zip containing at most one ONNX file per task:

task001.onnx
task002.onnx
...
task400.onnx
Note: if our evaluation metric requires adjustments—or, if we have to ban additional ONNX operators that compromise the aims of our contest—we will announce such changes and rescore submissions as needed.