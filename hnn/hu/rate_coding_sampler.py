# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.sampler import Sampler


class RateCodingSampler(Sampler):
    '''基于RateCoding的采样器

    通过encoder将一帧数据采样到window_size帧, 采样前可能会对数据进行正则化
    '''
    def __init__(self, window_size: int, encoder: torch.nn.Module) -> None:
        super(RateCodingSampler, self).__init__(window_size=window_size)
        self.encoder = encoder

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.size())
        shape.append(self.window_size)
        out = torch.zeros(shape)
        x = self.normalize(x)
        for i in range(self.window_size):
            out[..., i] = self.encoder(x)
        return out