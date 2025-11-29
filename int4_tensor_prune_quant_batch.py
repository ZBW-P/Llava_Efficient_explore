import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import bitsandbytes  # ensure bitsandbytes is installed

from image_utils import load_sample_image
from memory_utils import clear_memory
from benchmark_utils import run_all_stress_tests 

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAMPLE_PROMPT = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"


def INT4_TensorPrune_Quant_TRY_batch(use_topk=False):

    try:
        del model, processor
    except NameError:
        pass

    clear_memory()

    print("=" * 80)
    print("Method 1: INT4 + Tensor-wise KV Pruning + KV Quantization (Batch Test)")
    print("=" * 80)

    # 1) Bitsandbytes INT4 model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True, 
    )
    print("INT4 model loaded (bitsandbytes).\n")

    # Tensor-wise prune + quant  batch test
    batch_results = run_all_stress_tests(
        model,
        processor,
        maxbatchsize=32,
        method="tensorwise",
        kv_bits=8,
        pruning_ratio=0.7,
        use_kv_quant=True,
        use_prun=True,
        use_topk=use_topk, 
    )

    print("\nMethod 1 Batch Results (INT4 + Tensor-wise Prune + KV Quant):")
    for bs, r in batch_results.items():
        print(
            f"  BS={bs}: "
            f"Latency={r['total_time']:.3f}s, "
            f"Throughput={r['throughput']:.2f} tok/s, "
            f"Memory={r['peak_memory_gb']:.2f} GB"
        )
    print()


if __name__ == "__main__":
  print("\nUsing quantile for Pruning:")
  INT4_TensorPrune_Quant_TRY_batch()
  print("\nUsing Topk for Pruning:")
  INT4_TensorPrune_Quant_TRY_batch(True)
