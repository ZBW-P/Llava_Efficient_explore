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
## 5. Results and Observations

This section summarizes all experimental results across **NVIDIA L4** and **A100** GPUs.  
We separate the findings into **single-image inference**, **KV-based methods**,  
**batch inference**, and **quantization-related behavior**.

Full raw tables are available in the notebooks and test scripts.

---

### 5.1 Single-Image Inference (BS = 1)

#### **Baseline vs FlashAttention vs bitsandbytes**
| Method | L4 TPS | A100 TPS | Memory | Notes |
|--------|--------|-----------|--------|--------|
| Baseline | 15.07 | 22.05 | 13.52 GB | Default reference |
| FlashAttention 2 | 13.63 | 20.41 | 13.52 GB | Slight slowdown |
| bitsandbytes INT8/INT4 | 13–15 | ~15 | **4.69 GB** | Large memory ↓, speed ↓ |

➡ *FlashAttention gives minimal benefit; bitsandbytes saves memory but hurts speed.*

---

### 5.2 KV Cache & KV-Based Optimizations (BS = 1)

#### **KV Cache Baseline**
| GPU | TPS | Memory |
|------|-------|--------|
| L4 | **107.69 tok/s** | 13.55 GB |
| A100 | **161.41 tok/s** | 13.55 GB |

➡ **KV cache reuse provides ~7× speedup**, the largest non-pruning gain.

---

#### **Tensor-wise KV Pruning (Best for BS=1)**
| GPU | TPS | Memory |
|------|------|--------|
| L4 | **204.33** | 14.11 GB |
| A100 | **309.88** | 14.11 GB |

➡ **Fastest single-image method**, nearly doubling KV baseline throughput.

---

#### **Channel-wise KV Pruning (Stable, but weaker for BS=1)**
| Variant | L4 TPS | A100 TPS |
|---------|--------|-----------|
| Prune only | 130.82 | 195.93 |
| Quant only | 130.69 | 196.71 |

➡ Strong performance, but tensor-wise pruning is still the BS=1 winner.

---

#### **Model Fake Quantization (INT4 / INT8)**
| Method | L4 TPS | A100 TPS | Memory |
|--------|--------|-----------|---------|
| INT4 Tensor-wise | ~106 | ~104 | **5.05 GB** |
| INT4 Channel-wise | ~104 | ~105 | **5.05 GB** |

➡ **>60% memory savings**, with acceptable throughput loss.

---

### 5.3 Batch Inference (BS = 4, 8, 16, 32)

#### **Baseline Batch Throughput**
| GPU | Max TPS |
|------|---------|
| A100 | **3730 tok/s @ BS=32** |
| L4 | OOM at BS=32 |

---

#### **FlashAttention Batch**
- Always slower than baseline.
- Confirms LLaVA is **KV-cache bound**, not compute-bound.

---

#### **KV Cache Batch**
| GPU | Max TPS |
|------|----------|
| A100 | **4367.99 tok/s** |
| L4 | OOM @ BS=32 |

➡ KV cache improves throughput further as batch size increases.

---

### 5.4 Channel-Wise vs Tensor-Wise Pruning in Batch Mode

#### **Channel-Wise Pruning — Most Important Batch Finding**
(from `channel_prune_quant_batch.py`, prune-only)

| BS | L4 TPS | A100 TPS |
|----|--------|-----------|
| 4  | 361.28 | 1724.47 |
| 8  | OOM | 1234.43 |
| 16 | OOM | **6818.94** |
| 32 | OOM | 3457.34 |

#### Key Insights
- **6818.94 tok/s @ BS=16** (A100) is the **highest throughput in the entire project**.  
- Channel-wise pruning **outperforms KV cache baseline** in batch inference.  
- However, it becomes **unstable** when:
  - BS ≥ 32 on A100  
  - BS ≥ 8 on L4  
- This is expected because magnitude-based channel removal can break head balance.

➡ **Best method for large-batch, high-throughput production inference.**

---

#### **Tensor-Wise Pruning (Batch Mode)**
- Only BS=4 is stable.
- BS ≥ 8 → OOM on both L4 and A100.
- Throughput much lower than channel-wise pruning.

➡ **Best for BS=1 only; not suited for batching.**

---


### **INT4 / INT8 Quantization + Pruning (Batch Mode)**
General trends across `int4_*_batch.py` and `int8_*_batch.py`:

- Big **memory savings (30–70%)**.
- Significant **throughput drop** (30–70%).
- Many variations OOM at BS ≥ 16.
- Best when memory, not throughput, is the main constraint (e.g., L4).

➡ **Good memory-saving option; not competitive for speed.**

---

## 6. Summary

This project presents a practical evaluation of efficient inference techniques for LLaVA across NVIDIA L4 and A100 GPUs. Through systematic testing of baseline models, FlashAttention, bitsandbytes quantization, KV caching, KV pruning, and low-bit fake quantization, we measured their real performance impact in terms of latency, throughput, and memory usage. Our experiments span both single-image and large-batch inference, providing a complete view of optimization behavior in realistic settings.

Overall, the results show that **KV cache** is the dominant acceleration factor, **tensor-wise pruning** delivers the fastest BS=1 performance, **channel-wise pruning** provides the highest throughput for batch inference, and **INT4 KV quantization** achieves the most significant memory reduction with acceptable speed trade-offs. These findings offer a concise and actionable guideline for deploying efficient multimodal LLM inference under different hardware and workload constraints.

