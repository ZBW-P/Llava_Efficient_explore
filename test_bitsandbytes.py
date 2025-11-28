import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import bitsandbytes

from image_utils import load_sample_image_qnant
from memory_utils import clear_memory
from benchmark_utils import benchmark_model


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def Bit_bytes():
    clear_memory()
    print("Loading sample image...")
    image = load_sample_image_qnant()
    print(f"Image size: {image.size}\n")

    print("=" * 80)
    print("bitsandbytes Quantization 4 bits Test for LLaVA")
    print("=" * 80)

    # Load model in 4-bit using bitsandbytes
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("\nAWQ bitsandbytes quant 4 bit complete. Loading quantized model...\n")

    # Benchmark using same baseline benchmark tool
    results = benchmark_model(model, processor, image, SAMPLE_PROMPT)

    print("\nbitsandbytes Results:")
    print(f"  Avg Latency: {results['avg_total_time']:.3f} seconds")
    print(f"  Avg Throughput: {results['avg_throughput']:.2f} tokens/sec")
    print(f"  Avg Memory: {results['avg_memory_gb']:.2f} GB")
    print(f"\nSample Output:\n{results['sample_output'][:200]}...\n")


if __name__ == "__main__":
    Bit_bytes()
