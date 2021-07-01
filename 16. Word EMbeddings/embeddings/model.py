import logging 
import torch.nn as nn
from torch import Tensor

class W2VModel(nn.Module):
    def __init__(self, voc_size, emb_size) -> None:
        super(W2VModel, self).__init__()
        self.encoder = nn.Embedding(voc_size, emb_size)
        self.decoder = nn.Linear(emb_size, voc_size, bias=False)
        self.voc_size = voc_size
        self.emb_dim = emb_size
        self.initialize_weights()

    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.encoder(x)
        return self.decoder(out)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def initialize_weights(self):
        init_range = 0.5 / self.emb_dim
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(0, 0)
