# LLaVA Efficient Inference Exploration

This project benchmarks a collection of practical acceleration techniques for improving inference speed, throughput, and GPU memory efficiency of the multimodal model **llava-hf/llava-1.5-7b-hf** on NVIDIA **L4** and **A100** GPUs.

We focus on optimization methods that can be realistically deployed using HuggingFace Transformers, without relying on unsupported or custom CUDA kernels.  
Due to framework and architecture constraints, methods such as structural KV pruning and true quantization are not supported, so our work explores **masked pruning** and **fake quantization** to evaluate how much speedup is achievable in practice.

---

## 1. Project Description

Large model inference—especially multimodal LLMs like LLaVA—faces three key challenges:

- **High latency** during autoregressive decoding  
- **Large GPU memory consumption**, leading to OOM on mid-tier GPUs  
- **Low throughput**, especially for large batch sizes  

This project systematically evaluates the performance impact of:

- FlashAttention 2  
- bitsandbytes weight quantization (INT8/INT4)  
- KV cache baseline  
- Tensor-wise KV pruning  
- Channel-wise KV pruning  
- KV fake quantization (INT8/INT4)  
- Combined prune + quant approaches  

Our goal is to find the most effective techniques for **speeding up inference**, **reducing memory usage**, and **improving batch scalability**, while preserving generation quality.

---

## 2. Project Milestones and Completion Status

| Milestone | Description | Status |
|----------|-------------|--------|
| **1. Baseline Benchmark** | Measure default LLaVA inference on L4 & A100 | ✅ Completed |
| **2. Model Quantization** | bitsandbytes INT8/INT4 evaluation | ✅ Completed |
| **3. KV Cache – Part 1** | Implement and benchmark KV cache baseline | ✅ Completed |
| **4. KV Cache – Part 2** | Implement KV pruning + KV quant | ✅ Completed |
| **5. Testing & Comparison** | Full table benchmark: latency, throughput, memory | ✅ Completed |
| **6. Final Report** | Summary of results + conclusions | ✅ Completed |

All six milestones are now fully completed.  
Stress-test scenarios and large-batch evaluations have been integrated into the final analysis.

---

## 3. Repository Overview & Code Structure

The repository is organized into clean modules to support reproducible benchmarking of LLaVA acceleration techniques.  
The table below summarizes each major component:

| Folder/File | Description |
|-------------|-------------|
| `Experiment_A100.ipynb` | Full benchmark evaluation on NVIDIA A100 |
| `Experiment_L4.ipynb` | Full benchmark evaluation on NVIDIA L4 |
| `Experiment_Default.ipynb` | Baseline and comparison notebook |
| `test_baseline.py` | Baseline single-image (BS=1) inference |
| `test_baseline_batch.py` | Baseline batch-size inference |
| `test_flash_attn.py` | FlashAttention 2 benchmark |
| `test_flash_attn_batch.py` | FlashAttention 2 batch benchmark |
| `test_bitsandbytes.py` | INT8 / INT4 weight quantization |
| `test_bitsandbytes_batch.py` | bitsandbytes batch experiments |
| `test_kv.py` | KV cache baseline |
| `test_kv_prune.py` | Tensor-wise KV pruning |
| `test_kv_quant.py` | KV fake quantization |
| `test_kv_prune_quant.py` | Combined prune + quant |
| `*_batch.py` | Batch tests for BS = 4, 8, 16, 32 |
| `channel_prune_quant.py` | Channel-wise KV pruning and quantization |
| `int4_channel_prune_quant.py` | INT4 channel-wise prune/quant |
| `int4_tensor_prune_quant.py` | INT4 tensor-wise prune/quant |
| `int8_tensor_prune_quant.py` | INT8 tensor-wise prune/quant |
| `benchmark_utils.py` | Latency/throughput timers & GPU memory measurement |
| `memory_utils.py` | Peak memory tracker |
| `kv_utils.py` | KV cache extraction & pruning utilities(Unstructured) |
| `quant_utils.py` | Fake quantization operations |
| `image_utils.py` | Image preprocessing for LLaVA |

This structure ensures the full pipeline—from preprocessing to pruning to benchmarking—is modular and fully reproducible.

---

## 4. Example Commands to Execute the Code

All experiments in this repository can be executed using simple Python commands. Execution examples can be seen from three jupyter notebooks: Experiment_Default.ipynb Experiment_L4.ipynb Experiment_A100.ipynb

A. Baseline
```bash
python test_baseline.py
python test_baseline_batch.py
```
B. FlashAttention 2
```bash
python test_flash_attn.py
python test_flash_attn_batch.py
```
C. Bitsandbytes Quantization (INT8 / INT4)
```bash
python test_bitsandbytes.py
python test_bitsandbytes_batch.py
```
D. KV Cache Baseline
```bash
python test_kv.py
python test_kv_batch.py
```
E. KV Cache Pruning
Single-image
```bash
python test_kv_prune.py
python test_kv_prune_quant.py
python test_kv_quant.py
```
Batch experiments
```bash
python test_kv_prune_batch.py
python test_kv_prune_quant_batch.py
python test_kv_quant_batch.py
```
F. Tensor-wise Pruning / Quantization
INT8
```bash
python int8_tensor_prune_quant.py
python int8_tensor_prune_quant_batch.py
```
INT4
```bash
python int4_tensor_prune_quant.py
python int4_tensor_prune_quant_batch.py
```
G. Channel-wise Pruning / Quantization
INT4
```bash
python int4_channel_prune_quant.py
python int4_channel_prune_quant_batch.py
```
INT8
```bash
python int8_channel_prune_quant_batch.py
```
FP16 pruning
```bash
python channel_prune_quant.py
python channel_prune_quant_batch.py
```
H. AWQ (llmcompressor)
```bash
python test_llmcompressor_awq.py
python test_llmcompressor_awq_vllm.py
```
## 5. Results and observations 

