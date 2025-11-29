import torch
# ========================================================
# Original Element-Wise Pruning (Unstructured) save based on larger amount of zeroes
# ========================================================
def Pruning(kv_cache, pruning_ratio: float):
    """
    Original element-wise magnitude pruning (unstructured).
    Keep your original behavior unchanged.
    """
    pruned_cache = []
    for k, v in kv_cache:
        k_abs = k.float().abs()
        v_abs = v.float().abs()

        k_threshold = torch.quantile(k_abs, pruning_ratio)
        v_threshold = torch.quantile(v_abs, pruning_ratio)

        k_pruned = k.clone()
        k_pruned[k_abs < k_threshold] = 0

        v_pruned = v.clone()
        v_pruned[v_abs < v_threshold] = 0

        pruned_cache.append((k_pruned, v_pruned))

    return tuple(pruned_cache)


# ========================================================
# Channel-Wise Pruning save based on larger amount of zeroes
# ========================================================

def prune_kv_channels(kv_cache, pruning_ratio: float):
    pruned_cache = []

    for k, v in kv_cache:
        # k, v: [B, H, L, D]
        D = k.shape[-1]

        # Flatten across (batch * heads * seq_len) => [B*H*L, D]
        k_flat = k.view(-1, D).float()
        v_flat = v.view(-1, D).float()

        # L2 norm of each channel (D channels)
        k_norm = torch.norm(k_flat, dim=0)   # (D,)
        v_norm = torch.norm(v_flat, dim=0)   # (D,)

        # Combined magnitude importance per channel
        channel_importance = k_norm + v_norm

        # Compute pruning threshold
        threshold = torch.quantile(channel_importance, pruning_ratio)

        # Keep channels whose importance >= threshold
        keep_mask = (channel_importance >= threshold).to(k.device)  # shape (D,)

        # Reshape for broadcast: (1,1,1,D)
        mask = keep_mask[None, None, None, :]

        # Apply structured pruning
        k_pruned = k * mask
        v_pruned = v * mask

        pruned_cache.append((k_pruned, v_pruned))

    return tuple(pruned_cache)


########################################################
# Channel-Wise Pruning with Dimension Reduction different from above save in dimension
########################################################
def prune_kv_channels_structure(kv_cache, pruning_ratio: float):
    pruned_cache = []

    for k, v in kv_cache:
        D = k.shape[-1]

        k_flat = k.view(-1, D).float()
        v_flat = v.view(-1, D).float()

        k_norm = torch.norm(k_flat, dim=0)
        v_norm = torch.norm(v_flat, dim=0)

        importance = k_norm + v_norm

        threshold = torch.quantile(importance, pruning_ratio)
        keep_mask = importance >= threshold

        # True pruning: reduce D dimension
        k_pruned = k[:, :, :, keep_mask]
        v_pruned = v[:, :, :, keep_mask]

        pruned_cache.append((k_pruned, v_pruned))

    return tuple(pruned_cache)