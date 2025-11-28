from pyexpat import model
import time
from image_utils import load_images_from_urls
import torch

from memory_utils import clear_memory, get_memory_usage
from quant_utils import quantize_int8, dequantize_int8
from kv_utils import Pruning

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
    pruning=0.0
):
    clear_memory()

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    # PREFILL phase: obtain KV cache
    with torch.no_grad():
        output = model(**inputs, max_new_tokens=max_tokens, use_cache=True)
        kv_cache = output.past_key_values

    # KV PRUNING (if enabled)
    if use_prun:
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
    throughput = num_tokens / total_time
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
    pruning=0.0
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
            pruning=pruning
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
#   STRESS TEST (Sequence Length & Batch Size)
# ===========================================================

import numpy as np

def build_long_prompt(base_prompt, target_length):
    """Expand prompt to the target token-length by repeating."""
    multiplier = max(1, target_length // len(base_prompt.split()))
    long_prompt = " ".join([base_prompt] * multiplier)
    return long_prompt[:target_length]  # truncate to exact length


def stress_test_sequence_length(
    model,
    processor,
    image,
    base_prompt="Describe the image.",
    lengths=[128, 512, 2048, 4096],
    num_runs=1
):
    """Test latency & throughput under varying sequence lengths."""
    results = {}

    for L in lengths:
        long_prompt = build_long_prompt(base_prompt, L)
        print(f"\n--- Sequence Length = {L} ---")
        r = benchmark_model(model, processor, image, long_prompt, num_runs=num_runs)
        results[L] = r

    return results


def stress_test_batch_size(
    model,
    processor,
    images,
    prompts,
    batch_sizes=[1, 2, 4, 8],
    num_runs=1
):
    """
    Stress test for batch size.
    images: list of PIL images
    prompts: list of text prompts
    """
    results = {}

    for bs in batch_sizes:
        print(f"\n--- Batch Size = {bs} ---")

        # truncate or expand to match batch size
        batch_images = images[:bs] if len(images) >= bs else images * (bs // len(images) + 1)
        batch_images = batch_images[:bs]

        batch_prompts = prompts[:bs] if len(prompts) >= bs else prompts * (bs // len(prompts) + 1)
        batch_prompts = batch_prompts[:bs]

        # HuggingFace batch inference
        inputs = processor(
            text=batch_prompts,
            images=batch_images,
            return_tensors="pt"
        ).to(model.device)

        # Timing
        import time
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        end = time.time()

        total_time = end - start
        tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        throughput = (tokens * bs) / total_time

        results[bs] = {
            "latency": total_time,
            "throughput": throughput,
            "memory": get_memory_usage(),
        }

        print(f"BS={bs} latency={total_time:.2f}s throughput={throughput:.2f} tok/s mem={get_memory_usage():.2f} GB")

    return results


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
