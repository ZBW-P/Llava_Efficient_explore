import torch
import gc

def clear_memory():
    """ helper function to clear GPU memory """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def clean():
    """ delete global vars model/processor/inputs if exist """
    try:
        del model
        del processor
        del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except NameError:
        pass

def get_memory_usage():
    """ return max allocated GPU memory in GB """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0

def clear_gpu_memory():
    """ more aggressive GPU memory cleanup """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
