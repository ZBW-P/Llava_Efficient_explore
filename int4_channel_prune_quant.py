import time
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import bitsandbytes

from image_utils import load_sample_image
from memory_utils import clear_memory, get_memory_usage
from quant_utils import quantize_int8, dequantize_int8
from kv_utils import prune_kv_channels

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def run_inference_int4_channelwise(
    model,
    processor,
    image,
    prompt,
    max_tokens=100,
    pruning_ratio=0.3,
    use_kv_quant=True,
    kv_bits=8,
):

    clear_memory()

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    # 1) Prefill: 得到 KV cache
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    kv_cache = out.past_key_values

    # 2) Channel-wise pruning
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
    throughput = num_tokens / total_time
    peak_memory = get_memory_usage()

    return {
        "total_time": total_time,
        "num_tokens": num_tokens,
        "throughput": throughput,
        "peak_memory_gb": peak_memory,
        "output": decoded,
    }


def benchmark_method2_int4_channelwise(
    model,
    processor,
    image,
    prompt,
    num_runs=3,
    pruning_ratio=0.3,
    use_kv_quant=True,
    kv_bits=8,
):
    results = []

    print(f"Running {num_runs} inference passes for Method 2...")
    for i in range(num_runs):
        result = run_inference_int4_channelwise(
            model,
            processor,
            image,
            prompt,
            max_tokens=100,
            pruning_ratio=pruning_ratio,
            use_kv_quant=use_kv_quant,
            kv_bits=kv_bits,
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


def INT4_ChannelPrune_Quant_TRY():
    """
    Method 2:
    Bitsandbytes INT4 + channel-wise KV pruning + KV cache fake quantization.
    """
    print("Loading sample image...")
    image = load_sample_image()
    print(f"Image size: {image.size}")
    print()

    try:
        del model, processor
    except NameError:
        pass

    clear_memory()

    print("=" * 80)
    print("Method 2: INT4 + Channel-wise KV Pruning + KV Quantization")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    print("INT4 model loaded (bitsandbytes).")

    Method2_results = benchmark_method2_int4_channelwise(
        model,
        processor,
        image,
        SAMPLE_PROMPT,
        num_runs=3,
        pruning_ratio=0.3,
        use_kv_quant=True,
        kv_bits=8,
    )

    print("\nMethod 2 Results (INT4 + Channel-wise Prune + KV Quant):")
    print(f"  Average Latency: {Method2_results['avg_total_time']:.3f} seconds")
    print(f"  Average Throughput: {Method2_results['avg_throughput']:.2f} tokens/sec")
    print(f"  Average Memory: {Method2_results['avg_memory_gb']:.2f} GB")
    print(f"\nSample Output:\n{Method2_results['sample_output'][:200]}...\n")


if __name__ == "__main__":
    INT4_ChannelPrune_Quant_TRY()
