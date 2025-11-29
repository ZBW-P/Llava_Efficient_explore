import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import run_all_stress_tests

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def kv_TRY_batch():
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
    print("Baseline KV_cache Batch Test for LLaVA")
    print("=" * 80)

    # Load model and processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded.\n")

    # 🔥 Baseline KV cache batch test (no prune, no quant)
    batch_results = run_all_stress_tests(
        model,
        processor,
        maxbatchsize=32,
        method="tensorwise",
    )

    print("\nBaseline KV Cache Batch Results:")
    for bs, r in batch_results.items():
        print(
            f"  BS={bs}: "
            f"Latency={r['latency']:.3f}s, "
            f"Throughput={r['throughput']:.2f} tok/s, "
            f"Memory={r['memory']:.2f} GB"
        )
    print()


if __name__ == "__main__":
    kv_TRY_batch()
