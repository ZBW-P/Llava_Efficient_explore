import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import bitsandbytes

from image_utils import load_sample_image
from memory_utils import clear_memory
from int4_channel_prune_quant import (
    benchmark_method2_int4_channelwise,
    SAMPLE_PROMPT,
    MODEL_ID,
)


def Method3_INT4_ChannelPrune_Quant_Graph_TRY():
    """
    Method 3:
    Bitsandbytes INT4 + channel-wise KV pruning + KV quantization + graph/compiled execution.
    当前实现用 torch.compile 作为图执行的 placeholder。
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
    print("Method 3: INT4 + Channel-wise KV Pruning + KV Quant + Graph Execution")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    print("INT4 model loaded (bitsandbytes).")

    # 尝试图编译（PyTorch 2.x）
    try:
        model = torch.compile(model, mode="max-autotune")
        print("Model compiled with torch.compile for optimized execution.")
    except Exception as e:
        print(f"torch.compile failed, continue without compile: {e}")

    Method3_results = benchmark_method2_int4_channelwise(
        model,
        processor,
        image,
        SAMPLE_PROMPT,
        num_runs=3,
        pruning_ratio=0.3,
        use_kv_quant=True,
        kv_bits=8,
    )

    print("\nMethod 3 Results (INT4 + Channel-wise Prune + KV Quant + Graph):")
    print(f"  Average Latency: {Method3_results['avg_total_time']:.3f} seconds")
    print(f"  Average Throughput: {Method3_results['avg_throughput']:.2f} tokens/sec")
    print(f"  Average Memory: {Method3_results['avg_memory_gb']:.2f} GB")
    print(f"\nSample Output:\n{Method3_results['sample_output'][:200]}...\n")


if __name__ == "__main__":
    Method3_INT4_ChannelPrune_Quant_Graph_TRY()
