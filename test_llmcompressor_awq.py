import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests

from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

from image_utils import load_sample_image_qnant
from memory_utils import clear_memory, clean
from benchmark_utils import benchmark_model_quant


MODEL_ID = "llava-hf/llava-1.5-7b-hf"


# ----------------------------------------
#   Your Original AWQ Quantization Setup
# ----------------------------------------

def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def AWQ_TRY():
    DATASET_ID = "flickr30k"
    DATASET_SPLIT = "test"
    NUM_CALIBRATION_SAMPLES = 32
    MAX_SEQUENCE_LENGTH = 1024

    torch.cuda.empty_cache()
    clear_memory()

    print("=" * 80)
    print("AWQ Quantization Test for LLaVA")
    print("=" * 80)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.5),
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            sequential_targets=["LlamaDecoderLayer"],
            ignore=[
                "re:.*lm_head",
                "re:.*multi_modal_projector.*",
                "re:.*vision_tower.*"
            ],
        ),
    ]

    oneshot(
        model=model,
        tokenizer=MODEL_ID,
        dataset=DATASET_ID,
        splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        trust_remote_code_model=True,
        data_collator=data_collator,
        # sequential_targets=["LlamaDecoderLayer"],
        output_dir="llava-1.5-7b-INT8",
    )

    clean()


# ----------------------------------------
#   Run Benchmark After Quantization
# ----------------------------------------

def run_test(image):
    print("\nAWQ quantization complete. Loading quantized model...\n")

    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-1.5-7b-INT8",
        torch_dtype="auto",
    )

    dispatch_for_generation(model)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    results = benchmark_model_quant(model, processor, image)

    print("\nBaseline Results:")
    print(f"  Avg Latency: {results['avg_total_time']:.3f} seconds")
    print(f"  Avg Throughput: {results['avg_throughput']:.2f} tokens/sec")
    print(f"  Avg Memory: {results['avg_memory_gb']:.2f} GB")
    print(f"\nSample Output:\n{results['sample_output'][:200]}...\n")


# ----------------------------------------
#   Main Entry
# ----------------------------------------

if __name__ == "__main__":
    clean()
    print("Loading sample image...")
    image = load_sample_image_qnant()
    print(f"Image size: {image.size}\n")

    # If you want to run quantization:
    # AWQ_TRY()

    # Now run test on already-quantized model
    run_test(image)
