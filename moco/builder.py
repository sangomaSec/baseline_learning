from typing import Any

import torch
import torch.nn as nn


class MoCo(nn.Module):

    def __init__(self, base_encoder, momentum=0.999, t=0.07, dim=128, bs=65536):
        """
        :param base_encoder: query_encoder and key_encoder
        :param momentum: MoCo momentum of updating
        :param t: softmax temperature default is 0.07
        :param dim: the feature dimensions
        :param bs: batch size default is 65536
        """
        super(MoCo, self).__init__()

        self.dim = dim
        self.momentum = momentum
        self.t = t
        self.bs = bs

    @torch.no_grad
    def _enqueue_and_dequeue(self):
        """
        the function of the samples to dequeue and enqueue
        for that the encoder for the prior samples may be out date
        """

    @torch.no_grad
    def forward(self):
        """"""

    @torch.no_grad
    def _update_key_encoder(self):
        """"""

    @torch.no_grad
    def _shuffle_bn(self):
        """"""

