import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import bitsandbytes  # ensure bitsandbytes is installed

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import benchmark_model_q_p

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def INT8_TensorPrune_Quant_TRY():

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
    print("Method 1: INT8 + Tensor-wise KV Pruning + KV Quantization")
    print("=" * 80)

    # 1) Bitsandbytes INT8 
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_8bit=True,
        # torch_dtype=torch.float16,
    )
    print("INT8 model loaded (bitsandbytes).")

    Method1_results = benchmark_model_q_p(
        model,
        processor,
        image,
        SAMPLE_PROMPT,
        num_runs=3,
        use_kv_quant=True,
        kv_bits=8,
        use_prun=True,
        pruning=0.7,
    )

    print("\nMethod 1 Results (INT8 + Tensor-wise Prune + KV Quant):")
    print(f"  Average Latency: {Method1_results['avg_total_time']:.3f} seconds")
    print(f"  Average Throughput: {Method1_results['avg_throughput']:.2f} tokens/sec")
    print(f"  Average Memory: {Method1_results['avg_memory_gb']:.2f} GB")
    print(f"\nSample Output:\n{Method1_results['sample_output'][:200]}...\n")


if __name__ == "__main__":
    INT8_TensorPrune_Quant_TRY()
