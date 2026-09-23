"""
@author: Gaetan Hadjeres (Modified for PyTorch 1.10+)
"""

import torch


def cuda_variable(tensor):
    """Move tensor to CUDA if available."""
    if torch.cuda.is_available():
        # non_blocking=True: with pin_memory the transfer overlaps GPU compute.
        return tensor.cuda(non_blocking=True)
    return tensor


def to_numpy(variable):
    """Detach and move tensor to CPU numpy array."""
    if torch.cuda.is_available():
        return variable.detach().cpu().numpy()
    return variable.detach().numpy()


def init_hidden_state(rnn_type, num_layers, batch_size, hidden_size):
    """
    Zero initial hidden state. ``'lstm'`` -> ``(h_0, c_0)``, ``'gru'`` -> ``h_0`` only
    (``nn.GRU`` takes no tuple, hence passing the cell type in). zeros, not randn.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    if rnn_type == 'gru':
        return h
    c = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    return h, c


def init_hidden(num_layers, batch_size, lstm_hidden_size):
    """The LSTM spelling: returns all-zero (h_0, c_0). Called by `shared_trunk_model`."""
    return init_hidden_state('lstm', num_layers, batch_size, lstm_hidden_size)