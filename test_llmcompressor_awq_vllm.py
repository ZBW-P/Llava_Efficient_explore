import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
import time

from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

from image_utils import load_sample_image_qnant
from memory_utils import clear_memory, clean
from benchmark_utils import benchmark_model_quant

import base64

from io import BytesIO


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
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    recipe = [
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
        output_dir="llava-1.5-7b-INT8",
    )

    clean()


# ----------------------------------------
#   Run Benchmark After Quantization
# ----------------------------------------

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def run_inference_vllm(llm, processor, image, max_tokens=1000):
    """
    Run single inference with vLLM 0.6.1 multimodal API.
    """

    image_b64 = pil_to_base64(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_b64}"
                },
                {
                    "type": "text",
                    "text": "Please describe this image in detail.\n"
                }
            ],
        }
    ]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0
    )

    start_time = time.time()

    response = llm.chat(
        messages=messages,
        sampling_params=sampling_params
    )

    total_time = time.time() - start_time

    text = response[0].outputs[0].text
    num_tokens = response[0].outputs[0].token_count

    throughput = num_tokens / total_time
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "total_time": total_time,
        "num_tokens": num_tokens,
        "throughput": throughput,
        "peak_memory_gb": peak_memory,
        "output": text,
    }


def benchmark_vllm(llm, processor, image, num_runs=3):
    results = []

    print(f"Running {num_runs} inference passes...")
    for i in range(num_runs):
        result = run_inference_vllm(llm, processor, image)
        results.append(result)
        print(
            f" Run {i+1}/{num_runs}: "
            f"{result['throughput']:.2f} tokens/sec, "
            f"{result['peak_memory_gb']:.2f} GB"
        )

    avg_time = sum(r["total_time"] for r in results) / num_runs
    avg_throughput = sum(r["throughput"] for r in results) / num_runs
    avg_memory = sum(r["peak_memory_gb"] for r in results) / num_runs

    return {
        "avg_total_time": avg_time,
        "avg_throughput": avg_throughput,
        "avg_memory_gb": avg_memory,
        "sample_output": results[0]["output"],
    }
    
    
def LLM_inference(image):
  from vllm import LLM, SamplingParams
  print("\nAWQ quantization complete. Loading quantized model...\n")

  llm = LLM(MODEL_ID)

  processor = AutoProcessor.from_pretrained(MODEL_ID)

  results = benchmark_vllm(llm, processor, image)

  print("\nBaseline Results:")
  print(f"  Avg Latency: {results['avg_total_time']:.3f} seconds")
  print(f"  Avg Throughput: {results['avg_throughput']:.2f} tokens/sec")
  print(f"  Avg Memory: {results['avg_memory_gb']:.2f} GB")
  print(f"\nSample Output:\n{results['sample_output'][:200]}...\n")


# ----------------------------------------
#   Main Entry
# ----------------------------------------

if __name__ == "__main__":
    # clean()
    # print("Loading sample image...")
    # image = load_sample_image_qnant()
    # print(f"Image size: {image.size}\n")

    # If you want to run quantization:
    AWQ_TRY() #If already get compressed model, cite this

    # # Now run test on already-quantized model
    # LLM_inference(image)
