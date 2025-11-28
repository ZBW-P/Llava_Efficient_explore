import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import benchmark_model

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def FlashAttention_TRY():
    print("Loading sample image...")
    image = load_sample_image()
    print(f"Image size: {image.size}\n")

    # Cleanup
    try:
        del model, processor
    except:
        pass
    clear_memory()

    print("-" * 80)
    print("TEST 2: With Flash Attention 2")
    print("-" * 80)

    # Load model with FlashAttention 2
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Run benchmark
    flash_results = benchmark_model(model, processor, image, SAMPLE_PROMPT)

    print("\nFlash Attention 2 Results:")
    print(f"  Average Latency: {flash_results['avg_total_time']:.3f} seconds")
    print(f"  Average Throughput: {flash_results['avg_throughput']:.2f} tokens/sec")
    print(
        f"  Average Memory: {flash_results['avg_memory_gb']:.2f} GB"
    )
    print(f"\nSample Output:\n{flash_results['sample_output'][:200]}...\n")


if __name__ == "__main__":
    FlashAttention_TRY()
