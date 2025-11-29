import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import run_all_stress_tests

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def FlashAttention_TRY_batch():
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
    print("TEST 2: With Flash Attention 2 (Batch Test)")
    print("-" * 80)

    # Load model with FlashAttention 2
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Now run batch stress test instead of single-image benchmark
    batch_results = run_all_stress_tests(model, processor,maxbatchsize=32)

    print("\nFlash Attention 2 Batch Results:")
    for bs, r in batch_results.items():
        print(
            f"  BS={bs}: "
            f"Latency={r['latency']:.3f}s, "
            f"Throughput={r['throughput']:.2f} tok/s, "
            f"Memory={r['memory']:.2f} GB"
        )


if __name__ == "__main__":
    FlashAttention_TRY_batch()