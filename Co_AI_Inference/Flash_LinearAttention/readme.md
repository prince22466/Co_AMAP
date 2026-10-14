FlashAttention,This method delivers the exact mathematical result as standard attention but improves speed and memory usage by calculating the output matrix without ever creating the giant intermediate matrix in DRAM. It achieves this through two primary mechanisms:
Tiling( The algorithm breaks input matrices into blocks (or tiles) small enough to fit into the GPU's SRAM)
and kernel fusion(turns the entire attention operation, matrix multiplications, scaling, masking, and softmax, into a single GPU operation, or kernel.)
