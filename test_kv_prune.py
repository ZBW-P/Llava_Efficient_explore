import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import benchmark_model_q_p

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def prun_kv_TRY(URL=None,use_topk=False):
    if URL:
      print("Loading sample image...")
      image = load_sample_image(URL)
      print(f"Image size: {image.size}")
      print()
    else:
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
    print("Prun KV_cache Test for LLaVA")
    print("=" * 80)

    # Load model & processor
    clear_memory()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded.")

    # Apply magnitude pruning
    Prun_results = benchmark_model_q_p(
        model,
        processor,
        image,
        SAMPLE_PROMPT,
        num_runs=3,
        use_kv_quant=False,
        kv_bits=8,
        use_prun=True,
        pruning=0.7,
        use_topk=use_topk
    )

    print("\nPrun_kv_cacheresults Results:")
    print(f"  Average Latency: {Prun_results['avg_total_time']:.3f} seconds")
    print(f"  Average Throughput: {Prun_results['avg_throughput']:.2f} tokens/sec")
    print(f"  Average Memory: {Prun_results['avg_memory_gb']:.2f} GB")
    print(f"\nSample Output:\n{Prun_results['sample_output'][:200]}...\n")


if __name__ == "__main__":
    prun_kv_TRY()
    CAT_URL = "https://github.com/ZBW-P/Llava_Efficient_explore/blob/main/breed_abyssinian_cat.jpg?raw=1"
    prun_kv_TRY(CAT_URL)
    prun_kv_TRY(CAT_URL,use_topk=True)