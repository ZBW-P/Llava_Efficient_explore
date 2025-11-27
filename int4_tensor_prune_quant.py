import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import bitsandbytes  # ensure bitsandbytes is installed

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import benchmark_model_q_p

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def INT4_TensorPrune_Quant_TRY():
    """
    Method 1:
    Bitsandbytes INT4 + element-wise KV pruning (tensor-wise) + KV cache fake quantization.
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
    print("Method 1: INT4 + Tensor-wise KV Pruning + KV Quantization")
    print("=" * 80)

    # 1) Bitsandbytes INT4 模型
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    print("INT4 model loaded (bitsandbytes).")

    # 2) 直接复用你原来的 benchmark_model_q_p:
    #    - use_prun=True → 使用 element-wise Pruning
    #    - pruning=0.5   → 比较中等的剪枝比例（自己可改成 0.7 等）
    #    - use_kv_quant=True → 启用 KV fake INT 量化
    #    - kv_bits=8     → KV 用 8-bit (你可以自己改成 4)
    Method1_results = benchmark_model_q_p(
        model,
        processor,
        image,
        SAMPLE_PROMPT,
        num_runs=3,
        use_kv_quant=True,
        kv_bits=8,
        use_prun=True,
        pruning=0.5,
    )

    print("\nMethod 1 Results (INT4 + Tensor-wise Prune + KV Quant):")
    print(f"  Average Latency: {Method1_results['avg_total_time']:.3f} seconds")
    print(f"  Average Throughput: {Method1_results['avg_throughput']:.2f} tokens/sec")
    print(f"  Average Memory: {Method1_results['avg_memory_gb']:.2f} GB")
    print(f"\nSample Output:\n{Method1_results['sample_output'][:200]}...\n")


if __name__ == "__main__":
    INT4_TensorPrune_Quant_TRY()
