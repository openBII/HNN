# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.a2shu import A2SHU
from hnn.hu.rate_coding_sampler import RateCodingSampler
from typing import Callable


class A2SRateCoding(A2SHU):
    '''采样器为基于rate coding的采样器

    Args:
        window_size: 转换后的时间序列长度
        non_linear: 非线性函数
        precision_convert.converter: 精度转换函数
        sampler.encoder: 用于采样的编码器
    '''
    def __init__(self, window_size: int, encoder: torch.nn.Module,
                 converter: Callable[[torch.Tensor], torch.Tensor], non_linear: torch.nn.Module = None) -> None:
        super(A2SRateCoding, self).__init__(window_size, converter, non_linear)
        self.sampler = RateCodingSampler(
            window_size=self.window_size, encoder=encoder)
        self.check()