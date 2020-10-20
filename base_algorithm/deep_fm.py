import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as op


class DeepFM(nn.Module):
    """
    initialize the FM module
    """
    def __init__(self, k):
        self.k = k

    def forward(self):
        """"""
