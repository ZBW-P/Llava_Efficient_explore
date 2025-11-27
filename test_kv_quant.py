import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import benchmark_model_q_p

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def Quan_kv_TRY():
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
    print("Quantization KV_cache Test for LLaVA")
    print("=" * 80)

    # Load model
    clear_memory()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded.")

    # Apply INT8 fake KV cache quantization
    Quant_results = benchmark_model_q_p(
        model,
        processor,
        image,
        SAMPLE_PROMPT,
        num_runs=3,
        use_kv_quant=True,
        kv_bits=8,
        use_prun=False,
        pruning=0.0,
    )

    print("\nQuant_kv_cache Results:")
    print(f"  Average Latency: {Quant_results['avg_total_time']:.3f} seconds")
    print(f"  Average Throughput: {Quant_results['avg_throughput']:.2f} tokens/sec")
    print(f"  Average Memory: {Quant_results['avg_memory_gb']:.2f} GB")
    print(f"\nSample Output:\n{Quant_results['sample_output'][:200]}...\n")


if __name__ == "__main__":
    Quan_kv_TRY()
