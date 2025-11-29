import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import run_all_stress_tests

# Your original MODEL_ID and PROMPT kept here for baseline
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def Basic_TRY():
    print("Loading sample image...")
    image = load_sample_image()
    print(f"Image size: {image.size}")
    print()

    # Clear memory
    try:
        del model, processor
    except:
        pass
    clear_memory()

    print("=" * 80)
    print("Baseline_without_kv_cache Test for LLaVA")
    print("=" * 80)

    # Load processor & model
    clear_memory()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded.")

    # Baseline KV Cache test
    batch_results = run_all_stress_tests(model, processor,maxbatchsize=32)



if __name__ == "__main__":
    Basic_TRY()