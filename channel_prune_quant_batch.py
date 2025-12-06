import time
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import run_all_stress_tests

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def ChannelPrune_Quant_TRY_batch(use_structure_prun=False,use_kv_quant=True):
    """
    Method 2:
    FP16 LLava + channel-wise KV pruning + KV cache fake quantization (Batch Test).
    """
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
    print("Method 2: Channel-wise KV Pruning + KV Quantization (Batch Test)")
    print("=" * 80)

    # Load FP16 model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("LLava fp16 model loaded.\n")

    # Batch test using channelwise pipeline
    batch_results = run_all_stress_tests(
        model,
        processor,
        maxbatchsize=32,
        method="channelwise",
        kv_bits=8,
        pruning_ratio=0.7,
        use_kv_quant=use_kv_quant,
        use_prun=True,
        use_structure_prun=use_structure_prun,
    )

    print("\nMethod 2 Batch Results (Channel-wise Prune + KV Quant):")
    for bs, r in batch_results.items():
        print(
            f"  BS={bs}: "
            f"Latency={r['total_time']:.3f}s, "
            f"Throughput={r['throughput']:.2f} tok/s, "
            f"Memory={r['peak_memory_gb']:.2f} GB"
        )
    print()


if __name__ == "__main__":
  print("\nUsing unstructure for Pruning:")
  ChannelPrune_Quant_TRY_batch()
  print("\ndisable quantization for Pruning:")
  ChannelPrune_Quant_TRY_batch(use_kv_quant=False)

