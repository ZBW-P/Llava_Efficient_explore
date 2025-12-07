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

This section summarizes the performance of all acceleration methods tested on both **NVIDIA L4** and **A100** GPUs.  
To keep the README concise and readable, we provide **category-level result tables** and **clear observations**, while full raw tables remain available inside the notebook and test scripts.

---

### 5.1 Single-Image (BS = 1) Summary

#### **Baseline vs FlashAttention vs bitsandbytes**
| Method | L4 Throughput | A100 Throughput | Memory | Notes |
|--------|---------------|------------------|--------|-------|
| Baseline | 15 tok/s | 22 tok/s | 13.52 GB | Default reference |
| FlashAttention 2 | 13.6 tok/s | 20.4 tok/s | 13.52 GB | Slightly slower; compute-bound |
| bitsandbytes INT8/INT4 | ~14–15 tok/s | ~15 tok/s | **4.7 GB** | Memory ↓ but no speed gain |

➡ *FlashAttention shows limited benefit for LLaVA; bitsandbytes reduces memory but slows decoding.*

---

### 5.2 KV Cache and KV-Based Optimizations

#### **KV Cache Baseline**
| GPU | Throughput | Memory |
|-----|------------|--------|
| L4 | **107.7 tok/s** | 13.55 GB |
| A100 | **161.4 tok/s** | 13.55 GB |

➡ **KV cache reuse is the largest speedup factor—7× faster than baseline.**

---

#### **Tensor-wise KV Pruning**
| Variant | L4 TPS | A100 TPS | Memory |
|---------|--------|-----------|---------|
| Prune only | **204.3** | **309.9** | 14.11 GB |

➡ The **fastest BS=1 method overall**, doubling throughput vs KV baseline.

---

#### **Channel-wise KV Pruning**
| Variant | L4 TPS | A100 TPS | Memory |
|---------|--------|-----------|---------|
| Prune | 130.8 | 195.9 | 14.12 GB |
| Quant | 130.7 | 196.7 | 14.28 GB |

➡ Strong but slightly weaker than tensor-wise pruning for single-image inference.

---

#### **KV Quantization (INT4 / INT8 Fake Quant)**
| Method | L4 TPS | A100 TPS | Memory |
|--------|--------|-----------|---------|
| INT4 Tensor-wise | ~106 | ~104 | **5.05 GB** |
| INT4 Channel-wise | ~104 | ~105 | **5.05 GB** |

➡ **INT4 KV leads to >60% memory savings with minimal speed loss.**

---

### 5.3 Combined Prune + Quant

| Method | L4 TPS | A100 TPS | Notes |
|--------|--------|-----------|-------|
| Tensorwise prune+quant | ~362 | ~643 | BS=4 only |
| Quantile / Top-K prune+quant | Good TPS | OOM on L4 for large BS | Tradeoff between sparsity and memory |

➡ Hybrid prune+quant gives **balanced speed + memory benefit**.

---

### 5.4 Batch Inference Summary (BS = 4, 8, 16, 32)

#### **Baseline Batch**
| GPU | Max TPS |
|------|---------|
| A100 | **3730 tok/s @ BS=32** |
| L4 | OOM at BS=32 |

---

#### **FlashAttention Batch**
Throughput increases with batch size but remains **consistently below baseline**.

---

#### **KV Cache Batch**
| GPU | Max TPS |
|------|---------|
| A100 | **4368 tok/s** |
| L4 | OOM at BS=32 |

➡ KV cache again provides the largest gain.

---

#### **Channel-Wise INT4/INT8 Pruning**
| Method | L4 Max TPS | A100 Max TPS |
|---------|--------------|----------------|
| INT4 channel prune | 2843 | **2843** |
| INT8 channel prune | 1070 | 1070 |

➡ Channel pruning is **best for large-batch workloads**.

---

## 5.5 Summary of Main Results

#### ✅ **1. KV cache is the dominant factor influencing decoding speed.**  
Reusing KV boosts throughput from **15 → 160 tok/s** (L4) and **22 → 161 tok/s** (A100).  
No other method matches this jump.

---

#### ✅ **2. Different pruning strategies excel in different scenarios.**
- **Tensor-wise pruning** → best for **BS=1 / image-by-image inference**  
  (200–310 tok/s, highest speed of all methods)
- **Channel-wise pruning** → best for **batch inference (BS=16–32)**  
  (up to 2843 tok/s)

---

#### ✅ **3. Low-bit KV compression gives huge memory savings.**
- INT4 KV reduces memory from **13.5 GB → 5.0 GB**
- Throughput remains ~100 tok/s

➡ Ideal for deployment on memory-limited GPUs (L4 / T4 / consumer cards).

---

#### ✅ **4. Weight-only quantization is unsuitable for multimodal LLaVA.**
- bitsandbytes INT8/INT4 slows decoding  
- Throughput drops to 14–15 tok/s  
- Sometimes unstable generation

➡ KV-only quantization is far superior.

---

#### ✅ **5. FlashAttention provides limited benefit.**
- Slight latency gains  
- Does **not** significantly improve throughput  
- Compute-bound operations are not the bottleneck in LLaVA

---

#### ✅ **6. Hybrid KV prune + quant yields the most balanced real-world performance.**
- Strong throughput gains (300–640 TPS at BS=4)  
- Major memory savings  
- Works on both L4 and A100

➡ These methods are the most practical for real LLaVA deployments.

---

## 5.6 Final Observation

Across all experiments:

🔹 **KV cache** = biggest speedup  
🔹 **Tensor-wise prune** = fastest per-image  
🔹 **Channel-wise prune** = best for batching  
🔹 **INT4 KV** = best memory saving  
🔹 **bitsandbytes** = only useful for reducing model size, not speed  
🔹 **FlashAttention** = minor improvement  
🔹 **Hybrid prune + quant** = best overall tradeoff

This provides a clear roadmap for practitioners building efficient LLaVA inference systems.


