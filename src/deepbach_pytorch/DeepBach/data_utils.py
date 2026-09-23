"""
@author: Gaetan Hadjeres
"""

import torch


def mask_entry(tensor, entry_index, dim):
    """Drop entry entry_index along dim; the index tensor is built on the input's device."""
    idx = [i for i in range(tensor.size(dim)) if i != entry_index]
    idx = torch.tensor(idx, dtype=torch.long, device=tensor.device)
    return tensor.index_select(dim, idx)


def reverse_tensor(tensor, dim):
    """Reverse along dim; the index is built on the input's device, as in mask_entry."""
    idx = torch.arange(tensor.size(dim) - 1, -1, -1,
                       dtype=torch.long, device=tensor.device)
    return tensor.index_select(dim, idx)