Dataset Description
The objective of this competition is to create a suite of neural networks to implement a variety of transformations, where each transformation is implicitly described by a series of image grids. For example, the example pairs for one task might demonstrate the concept of rotation, whereas another might involve cropping and/or magnification. Your network for a given task should not only achieve the desired result across all exemplars, but also do so using the simplest possible architecture.

Task files(in taskfiles folder)
The information for each of the four-hundred tasks is stored in an appropriately named json file (e.g., task001.json, task002.json). The file for a given task contains a dictionary with three fields:

"train": a list of input/output pairs originally included in ARC-AGI-1 for training
"test": a list of input/output pairs originally included in ARC-AGI-1 for testing
"arc-gen": a list of additional input/output pairs included in the ARC-GEN-100K dataset
A "pair" is a dictionary with two fields:

"input": the input "grid" for the pair.
"output": the output "grid" for the pair.
A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 1x1 and the largest is 30x30. Before being passed into your networks, each input grid will be converted into a tensor of size [BATCH_DIM=1, CHANNELS=10, HEIGHT=30, WIDTH=30], using a one-hot channel encoding for each colored pixel, and a zero-hot channel encoding for any "clear" pixels that lie outside the original border.

For all pairs in each of the example subsets (i.e., "train" +"test" + "arc-gen"), your submitted network should successfully construct the output grid(s) corresponding to the input grid(s). "Constructing the output grid" involves filling each cell in the grid with a 1 for the correct channel and 0 for others (or, a 0 for all channels if this cell lies beyond the image border). Only exact solutions (where all cells match the expected answer) can be said to be correct.

In addition, our official scoring metric will also employ a private dataset (containing a smaller number of examples per task) when validating these networks, so as to prevent overfitting.