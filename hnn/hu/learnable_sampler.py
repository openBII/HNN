# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.sampler import Sampler


class LearnableSampler(Sampler):
    '''可学习的采样器

    将输入数据看作一帧数据, 通过线性变换采样到window_size帧
    '''
    def __init__(self, window_size: int) -> None:
        super(LearnableSampler, self).__init__(window_size=window_size)
        self.linear = torch.nn.Linear(1, self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.linear(x)
        return x