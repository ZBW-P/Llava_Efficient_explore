import time
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from image_utils import load_sample_image
from memory_utils import clear_memory, get_memory_usage
from quant_utils import quantize_int8, dequantize_int8
from kv_utils import prune_kv_channels
from benchmark_utils import benchmark_channelwise

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def ChannelPrune_Quant_TRY():
    """
    Method 2:
    Bitsandbytes channel-wise KV pruning + KV cache fake quantization.
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
    print("Method 2: Channel-wise KV Pruning + KV Quantization")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("LLava model loaded .")

    Method2_results = benchmark_channelwise(
        model,
        processor,
        image,
        SAMPLE_PROMPT,
        num_runs=3,
        pruning_ratio=0.3,
        use_kv_quant=True,
        kv_bits=8,
    )

    print("\nMethod 2 Results (Channel-wise Prune + KV Quant):")
    print(f"  Average Latency: {Method2_results['avg_total_time']:.3f} seconds")
    print(f"  Average Throughput: {Method2_results['avg_throughput']:.2f} tokens/sec")
    print(f"  Average Memory: {Method2_results['avg_memory_gb']:.2f} GB")
    print(f"\nSample Output:\n{Method2_results['sample_output'][:200]}...\n")


if __name__ == "__main__":
    ChannelPrune_Quant_TRY()
