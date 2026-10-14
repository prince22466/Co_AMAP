Link: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287


New Gemma 4 Models Available: 12B, E4B, and E2B
Hi everyone,

We've added support for several smaller Gemma 4 variants. You can now use any of the following models for your agents:

gemma-4-31b-it-qat-w4a16-ct
gemma-4-12b-it-qat-w4a16-ct
gemma-4-e4b-it-qat-w4a16-ct
gemma-4-e2b-it-qat-w4a16-ct
We don't support mixing variants within a submission, however. Every agent and subagent in your submission must use the same variant (though they can attach different LoRA adapters.)

These smaller models open up some new strategies:

More test time compute: Inference is faster and memory burden lower so you can execute more and longer reasoning turns and iteration cycles. You could also experiment with using ParallelAgent for concurrent LLM calls (though tool calls are still executed sequentially).
Faster and more accessible LoRA finetuning: Experiment with different setups more quickly and on more accessible GPUs. We provide L4x4s for use in interactive notebooks, but adapters can now be finetuned on a wider variety of consumer hardware.
More specialized agents: Smaller models use smaller LoRA adapters, so you can include more of them in your submission.
We're excited to see what you build!