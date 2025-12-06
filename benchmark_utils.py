from pyexpat import model
import time
from image_utils import load_images_from_urls,load_sample_image
import torch
from transformers.cache_utils import DynamicCache
from memory_utils import clear_memory, get_memory_usage
from quant_utils import quantize_int8, dequantize_int8
from kv_utils import Pruning,Pruning_topk
from kv_utils import prune_kv_channels

# -----------------------------------------------------------
#   BASELINE / FLASH-ATTEINON MODEL INFERENCE
# -----------------------------------------------------------

def run_inference(model, processor, image, prompt, max_tokens=100):
    """ run single inference and return metrics """

    clear_memory()

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    end_time = time.time()
    total_time = end_time - start_time

    decoded = processor.decode(output[0], skip_special_tokens=True)
    num_tokens = output.shape[1] - inputs['input_ids'].shape[1]
    throughput = num_tokens / total_time
    peak_memory = get_memory_usage()

    return {
        'total_time': total_time,
        'num_tokens': num_tokens,
        'throughput': throughput,
        'peak_memory_gb': peak_memory,
        'output': decoded
    }


def benchmark_model(model, processor, image, prompt, num_runs=3):
    results = []

    print(f"Running {num_runs} inference passes...")
    for i in range(num_runs):
        result = run_inference(model, processor, image, prompt)
        results.append(result)
        print(
            f" Run {i+1}/{num_runs}: "
            f"{result['throughput']:.2f} tokens/sec, "
            f"{result['peak_memory_gb']:.2f} GB"
        )

    avg_time = sum(r['total_time'] for r in results) / num_runs
    avg_throughput = sum(r['throughput'] for r in results) / num_runs
    avg_memory = sum(r['peak_memory_gb'] for r in results) / num_runs

    return {
        'avg_total_time': avg_time,
        'avg_throughput': avg_throughput,
        'avg_memory_gb': avg_memory,
        'sample_output': results[0]['output']
    }

# -----------------------------------------------------------
#   QUANTIZED (MODEL QUANT) INFERENCE
# -----------------------------------------------------------

def run_inference_quant(model, processor, image, max_tokens=100):
    """ run single inference and return metrics """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please detail describe this image\n"},
                {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    end_time = time.time()
    total_time = end_time - start_time

    decoded = processor.decode(output[0], skip_special_tokens=True)
    num_tokens = output.shape[1] - inputs['input_ids'].shape[1]
    throughput = num_tokens / total_time
    peak_memory = get_memory_usage()

    return {
        'total_time': total_time,
        'num_tokens': num_tokens,
        'throughput': throughput,
        'peak_memory_gb': peak_memory,
        'output': decoded
    }


def benchmark_model_quant(model, processor, image, num_runs=3):
    results = []

    print(f"Running {num_runs} inference passes...")
    for i in range(num_runs):
        result = run_inference_quant(model, processor, image)
        results.append(result)
        print(
            f" Run {i+1}/{num_runs}: "
            f"{result['throughput']:.2f} tokens/sec, "
            f"{result['peak_memory_gb']:.2f} GB"
        )

    avg_time = sum(r['total_time'] for r in results) / num_runs
    avg_throughput = sum(r['throughput'] for r in results) / num_runs
    avg_memory = sum(r['peak_memory_gb'] for r in results) / num_runs

    return {
        'avg_total_time': avg_time,
        'avg_throughput': avg_throughput,
        'avg_memory_gb': avg_memory,
        'sample_output': results[0]['output']
    }

# -----------------------------------------------------------
#   KV CACHE PRUNE / KV CACHE QUANT / MIXED TEST
# -----------------------------------------------------------


def run_inference_quan_prun(
    model,
    processor,
    image,
    prompt,
    max_tokens=100,
    use_kv_quant=False,
    kv_bits=8,
    use_prun=False,
    pruning=0.0,
    bs=1,
    use_topk=False,
    image_prun=False,
):
    clear_memory()
    if not isinstance(image, (list, tuple)):
        image = [image]

    texts = [prompt] * len(image)

    inputs = processor(text=texts, images=image, return_tensors="pt").to(model.device)

    # PREFILL phase: obtain KV cache
    with torch.no_grad():
        output = model(**inputs, max_new_tokens=max_tokens, use_cache=True)
    kv_cache = output.past_key_values
    
    # KV PRUNING (if enabled)
    if use_prun:
      if use_topk:
        kv_cache = Pruning_topk(kv_cache, pruning_ratio=pruning)
      else:
        kv_cache = Pruning(kv_cache, pruning_ratio=pruning)

    # KV QUANTIZATION (if enabled)
    if use_kv_quant:
        q_cache = []
        for k, v in kv_cache:
            k_q, k_scale = quantize_int8(k, bits=kv_bits)
            v_q, v_scale = quantize_int8(v, bits=kv_bits)
            q_cache.append(((k_q, k_scale), (v_q, v_scale)))
        kv_cache = tuple(q_cache)

    # DEQUANT
    if use_kv_quant:
        deq_cache = []
        for (k_q, k_scale), (v_q, v_scale) in kv_cache:
            k_fp = dequantize_int8(k_q, k_scale).to(torch.float16)
            v_fp = dequantize_int8(v_q, v_scale).to(torch.float16)
            deq_cache.append((k_fp, v_fp))
        kv_cache = tuple(deq_cache)

    kv_length = kv_cache[0][0].shape[2]
    cache_position = torch.arange(kv_length, kv_length + 1).to(model.device)

    # GENERATION using pruned/quantized KV
    start_time = time.time()

    gen_out = model.generate(
      **inputs,
      max_new_tokens=max_tokens,
      use_cache=True,
      past_key_values=kv_cache,
      cache_position=cache_position,
    )
    end_time = time.time()

    total_time = end_time - start_time
    decoded = processor.decode(gen_out[0], skip_special_tokens=True)
    num_tokens = gen_out.shape[1] - 1
    throughput = num_tokens*bs / total_time
    peak_memory = get_memory_usage()

    return {
        'total_time': total_time,
        'num_tokens': num_tokens,
        'throughput': throughput,
        'peak_memory_gb': peak_memory,
        'output': decoded
    }


def benchmark_model_q_p(
    model,
    processor,
    image,
    prompt,
    num_runs=3,
    use_kv_quant=False,
    kv_bits=8,
    use_prun=False,
    pruning=0.0,
    use_topk=False
):
    results = []

    print(f"Running {num_runs} inference passes...")

    for i in range(num_runs):
        result = run_inference_quan_prun(
            model,
            processor,
            image,
            prompt,
            use_kv_quant=use_kv_quant,
            kv_bits=kv_bits,
            use_prun=use_prun,
            pruning=pruning,
            bs=1,
            use_topk=use_topk
        )
        results.append(result)

        print(
            f" Run {i+1}/{num_runs}: "
            f"{result['throughput']:.2f} tokens/sec, "
            f"{result['peak_memory_gb']:.2f} GB"
        )

    avg_time = sum(r['total_time'] for r in results) / num_runs
    avg_throughput = sum(r['throughput'] for r in results) / num_runs
    avg_memory = sum(r['peak_memory_gb'] for r in results) / num_runs

    return {
        'avg_total_time': avg_time,
        'avg_throughput': avg_throughput,
        'avg_memory_gb': avg_memory,
        'sample_output': results[0]['output']
    }
    
# ===========================================================
#   Channel_WISE KV PRUNING + KV QUANTIZATION
# ===========================================================    
from kv_utils import prune_kv_channels
    
def run_inference_channelwise(
    model,
    processor,
    image,
    prompt,
    max_tokens=100,
    pruning_ratio=0.3,
    use_kv_quant=True,
    kv_bits=8,
    use_structure_prun=False,
    bs=1
):

    clear_memory()
    if not isinstance(image, (list, tuple)):
        image = [image]

    texts = [prompt] * len(image)

    inputs = processor(text=texts, images=image, return_tensors="pt").to(model.device)

    # 1) Prefill: get KV cache
    with torch.no_grad():
        out = model(**inputs,max_new_tokens=max_tokens, use_cache=True)
    kv_cache = out.past_key_values

    # 2) Channel-wise pruning
    if use_structure_prun:
        kv_cache = kv_cache
    else:
        kv_cache = prune_kv_channels(kv_cache, pruning_ratio=pruning_ratio)
    
    # 3) KV fake quantization
    if use_kv_quant:
        q_cache = []
        for k, v in kv_cache:
            k_q, k_scale = quantize_int8(k, bits=kv_bits)
            v_q, v_scale = quantize_int8(v, bits=kv_bits)
            q_cache.append(((k_q, k_scale), (v_q, v_scale)))
        kv_cache = tuple(q_cache)
    if use_kv_quant:
        deq_cache = []
        for (k_q, k_scale), (v_q, v_scale) in kv_cache:
            k_fp = dequantize_int8(k_q, k_scale).to(torch.float16)
            v_fp = dequantize_int8(v_q, v_scale).to(torch.float16)
            deq_cache.append((k_fp, v_fp))
        kv_cache = tuple(deq_cache)

    kv_length = kv_cache[0][0].shape[2]
    cache_position = torch.arange(kv_length, kv_length + 1).to(model.device)

    start_time = time.time()
    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        past_key_values=kv_cache,
        cache_position=cache_position,
    )
    end_time = time.time()

    total_time = end_time - start_time
    decoded = processor.decode(gen_out[0], skip_special_tokens=True)
    num_tokens = gen_out.shape[1] - 1
    throughput = num_tokens*bs / total_time
    peak_memory = get_memory_usage()

    return {
        "total_time": total_time,
        "num_tokens": num_tokens,
        "throughput": throughput,
        "peak_memory_gb": peak_memory,
        "output": decoded,
    }


def benchmark_channelwise(
    model,
    processor,
    image,
    prompt,
    num_runs=3,
    pruning_ratio=0.3,
    use_kv_quant=True,
    kv_bits=8,
    use_structure_prun=False,
):
    results = []

    print(f"Running {num_runs} inference passes for Method 2...")
    for i in range(num_runs):
        result = run_inference_channelwise(
            model,
            processor,
            image,
            prompt,
            max_tokens=100,
            pruning_ratio=pruning_ratio,
            use_kv_quant=use_kv_quant,
            kv_bits=kv_bits,
            use_structure_prun=use_structure_prun,
            bs=1
        )
        results.append(result)
        print(
            f" Run {i+1}/{num_runs}: "
            f"{result['throughput']:.2f} tokens/sec, "
            f"{result['peak_memory_gb']:.2f} GB"
        )

    avg_time = sum(r["total_time"] for r in results) / num_runs
    avg_throughput = sum(r["throughput"] for r in results) / num_runs
    avg_memory = sum(r["peak_memory_gb"] for r in results) / num_runs

    return {
        "avg_total_time": avg_time,
        "avg_throughput": avg_throughput,
        "avg_memory_gb": avg_memory,
        "sample_output": results[0]["output"],
    }


# ============================================================
#   Batch Size Stress Test (Multi-image test)
# ============================================================

def stress_test_batch_size_basic(
    model,
    processor,
    images,
    prompt="USER: <image>\nDescribe this image in detail.\nASSISTANT:",
    batch_sizes=(4, 8, 16, 32, 64, 128),
    max_tokens=100,
):
    print("\n==============================")
    print("  Batch Size Stress Test (true batch forward)")
    print("==============================")

    results = {}

    max_bs = min(max(batch_sizes), len(images))
    batch_sizes = [bs for bs in batch_sizes if bs <= max_bs]

    for bs in batch_sizes:
        print(f"\n--- Batch Size = {bs} (1 forward pass with {bs} images) ---")

        # Slice images for this batch
        batch_imgs = images[:bs]

        clear_memory()
        try:

          inputs = processor(
              text=[prompt] * bs,
              images=batch_imgs,
              return_tensors="pt",
              padding=True,
          ).to(model.device)

          start_time = time.time()

          with torch.no_grad():
              gen_out = model.generate(
                  **inputs,
                  max_new_tokens=max_tokens,
                  use_cache=True
              )

          end_time = time.time()

          total_time = end_time - start_time
          num_tokens = gen_out.shape[1] - 1
          throughput = (num_tokens * bs) / total_time
          peak_memory = get_memory_usage()

          results[bs] = {
              "latency": total_time,
              "throughput": throughput,
              "memory": peak_memory,
          }

          print(
              f" ==> BS={bs}: latency={total_time:.3f}s, "
              f"throughput={throughput:.2f} tok/s, "
              f"max mem={peak_memory:.2f} GB"
          )
        except Exception as e:
          print(f"[ERROR] BS={bs} failed: {e}")
          clear_memory()

    return results


def stress_test_batch_size(
    model,
    processor,
    images,
    prompt="USER: <image>\nDescribe this image.\nASSISTANT:",
    batch_sizes=(4, 8, 16, 32, 64, 128),
    method="tensorwise", 
    kv_bits=8,
    pruning_ratio=0.3,
    use_kv_quant=True,
    use_prun=True,
    max_tokens=100,
    use_structure_prun=False,
    use_topk=False
):
    print("\n==============================")
    print(f"  Batch Size Stress Test ({method})")
    print("==============================")

    results = {}

    # auto-filter batch sizes
    max_bs = min(max(batch_sizes), len(images))
    batch_sizes = [b for b in batch_sizes if b <= max_bs]

    for bs in batch_sizes:
        print(f"\n--- Batch Size = {bs} ---")

        batch_imgs = images[:bs]
        clear_memory()
        try:
          if method == "tensorwise":
              r = run_inference_quan_prun(
                  model, processor,
                  batch_imgs, prompt,
                  max_tokens=max_tokens,
                  use_kv_quant=use_kv_quant,
                  kv_bits=kv_bits,
                  use_prun=use_prun,
                  pruning=pruning_ratio,
                  bs=bs,
                  use_topk=use_topk  
              )
          elif method == "channelwise":
              r = run_inference_channelwise(
                  model, processor,
                  batch_imgs, prompt,
                  max_tokens=max_tokens,
                  pruning_ratio=pruning_ratio,
                  use_kv_quant=use_kv_quant,
                  kv_bits=kv_bits,
                  use_structure_prun=use_structure_prun,
                  bs=bs
              )
          else:
              raise ValueError("method must be tensorwise or channelwise")

          results[bs] = r

          print(
              f" ==> BS={bs}, "
              f"latency={r['total_time']:.3f}s, "
              f"throughput={r['throughput']:.2f} tok/s, "
              f"max_mem={r['peak_memory_gb']:.2f} GB"
          )
        except Exception as e:
          print(f"[ERROR] BS={bs} failed: {e}")
          clear_memory()

    return results

# ============================================================
#   Unified Function: Run Both Tests
# ============================================================

def run_all_stress_tests(model, processor,maxbatchsize=32,method=None,kv_bits=8,pruning_ratio=0.7,use_kv_quant=False,use_prun=False,use_structure_prun=False,use_topk=False):
    clear_memory()
    
    HF_TEST_IMAGES = [
        "https://llava-vl.github.io/static/images/view.jpg"
    ]*maxbatchsize
    images=load_images_from_urls(HF_TEST_IMAGES)
    if method in ["tensorwise","channelwise"]:
        batch_results = stress_test_batch_size(model, processor, images,
                                               method=method,kv_bits=kv_bits,
                                               pruning_ratio=pruning_ratio,
                                               use_kv_quant=use_kv_quant,batch_sizes=(4, 8, 16, 32, 64, 128),
                                               use_prun=use_prun,use_structure_prun=use_structure_prun,
                                               use_topk=use_topk
                                               )
    else:
        batch_results = stress_test_batch_size_basic(model, processor, images)

    return batch_results


# ===========================================================
#   ACCURACY TEST (Multi-image evaluation)
# ===========================================================

def accuracy_test(
    model,
    processor,
    image_list,
    method_name="Unknown",
    max_tokens=80
):
    """
    Run accuracy test for a list of images.
    Returns captions for later manual comparison.
    """
    results = []

    print(f"\n=== Accuracy Test: {method_name} ===")

    for idx, img in enumerate(image_list):
        prompt = "Describe this image in detail."
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_tokens)
        
        caption = processor.decode(output[0], skip_special_tokens=True)
        print(f"[Image {idx}] {caption[:100]}...")

        results.append({
            "image_id": idx,
            "caption": caption
        })

    return {
        "method": method_name,
        "captions": results
    }
