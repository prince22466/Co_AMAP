# Inference Engines and Model Formats

> **Knowledge verified through:** August 29, 2026

An **inference engine** or **runtime** executes a trained model. A **model format** stores the model graph, weights, configuration, and metadata. They are related, but they are not interchangeable concepts.

## Inference engines and runtimes

### TensorRT

[NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) is an SDK for optimizing and running neural-network inference on NVIDIA GPUs. It compiles a trained model into a serialized TensorRT engine (also called a plan file) for a target deployment environment.

TensorRT engines are optimized deployment artifacts, not portable model archives. By default, their compatibility is tied to the TensorRT version, operating system, CPU architecture, and GPU model used to build them. TensorRT provides compatibility options, but they must be configured explicitly.

### TensorRT-LLM

[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) is NVIDIA's toolkit for optimized LLM inference and serving on NVIDIA GPUs.

As of TensorRT-LLM 1.2, PyTorch is its sole execution backend. The former TensorRT engine backend and the `trtllm-build` workflow have been removed. TensorRT-LLM now loads supported Hugging Face checkpoints directly, so current TensorRT-LLM deployments should not be described as using prebuilt TensorRT engines.

This change applies to **TensorRT-LLM**. Standalone **TensorRT** still builds and executes TensorRT engines for general neural-network inference.

### ONNX Runtime

[ONNX Runtime](https://onnxruntime.ai/docs/) is a cross-platform runtime for ONNX models. It supports CPUs, GPUs, and specialized accelerators through Execution Providers such as CUDA, TensorRT, OpenVINO, DirectML, CoreML, and Qualcomm QNN.

When multiple Execution Providers are configured, ONNX Runtime can partition a model graph into supported subgraphs and assign them to different providers, with a lower-priority provider such as the CPU acting as a fallback.

### vLLM

[vLLM](https://docs.vllm.ai/) is an inference and serving engine focused on high-throughput generative-model workloads. It provides online serving APIs, including an OpenAI-compatible HTTP server, and supports multiple accelerator platforms.

For normal deployments, vLLM commonly loads Hugging Face models with Safetensors weights. GGUF support is available through the `vllm-gguf-plugin`, but the vLLM documentation currently describes it as experimental and under-optimized.

### llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a lightweight C/C++ inference runtime centered on GGUF models. It supports CPU execution and numerous acceleration backends, including CUDA, Metal, HIP, Vulkan, SYCL, and others.

It is well suited to local, desktop, edge, and resource-constrained deployments, but it is not limited to laptops or CPU-only inference.

## Model and deployment formats

### GGUF

[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) is a binary model format designed for GGML-based runtimes such as llama.cpp. It stores tensors and the metadata needed to load a model and is designed for efficient, often single-file distribution and loading.

GGUF is not itself a compression method. A GGUF file may contain unquantized or quantized tensors. Quantized GGUF models are popular for inference performance.

### Hugging Face checkpoints and Safetensors

Hugging Face model repositories commonly store configuration, tokenizer files, and model weights in one or more Safetensors files. This is a primary input format for runtimes such as vLLM and the current TensorRT-LLM PyTorch backend.

Safetensors stores tensors safely and efficiently, while the surrounding Hugging Face repository supplies the model configuration and other required assets. It is therefore usually a checkpoint layout rather than a single self-contained model file.

### ONNX

[ONNX](https://onnx.ai/) is a standardized model interchange format that represents a computation graph, operators, weights, and metadata. It is useful for moving supported models between frameworks and deployment runtimes, including ONNX Runtime and TensorRT.

ONNX improves portability, but it does not guarantee indefinite archival compatibility. Deployment still depends on the ONNX opset, runtime operator support, custom operators, and any external weight data.

### TensorRT engines

A TensorRT engine is a compiled, serialized artifact produced by standalone TensorRT for fast inference on a target NVIDIA deployment environment. It is appropriate when the target hardware and software stack are known and controlled.

Because an engine is hardware- and version-sensitive by default, retain the source model or checkpoint and the reproducible build configuration rather than treating the engine as the canonical long-term model copy.

## Selection guide

| Use case | Typical runtime | Typical model format | Main considerations |
| --- | --- | --- | --- |
| Local or edge LLM inference | llama.cpp | GGUF, often quantized | Low memory use, broad CPU/GPU backend support, and simple distribution |
| High-throughput generative-model serving | vLLM | Hugging Face checkpoint, usually Safetensors | Continuous serving, accelerator support, batching, and API compatibility |
| NVIDIA-focused LLM serving | TensorRT-LLM | Supported Hugging Face checkpoint | NVIDIA-specific LLM optimizations using the current PyTorch backend |
| Portable inference across different hardware | ONNX Runtime | ONNX | Execution Provider coverage, operator support, and graph partitioning |
| General neural-network inference on a fixed NVIDIA stack | TensorRT | TensorRT engine built from ONNX or another supported input | Strong target-specific optimization but limited artifact portability |

No runtime is universally the fastest. Compare candidates using the actual model, precision or quantization method, prompt and output lengths, batch size, concurrency, latency target, throughput target, and deployment hardware.

## References

- [TensorRT: How TensorRT Works](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html)
- [TensorRT-LLM: TensorRT Backend Removed](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [ONNX Runtime Architecture](https://onnxruntime.ai/docs/reference/high-level-design.html)
- [vLLM Online Serving](https://docs.vllm.ai/en/latest/serving/online_serving/)
- [vLLM GGUF Support](https://docs.vllm.ai/en/stable/features/quantization/gguf/)
- [llama.cpp Documentation](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
