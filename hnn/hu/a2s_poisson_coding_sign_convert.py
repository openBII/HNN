# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.a2shu import A2SHU
from hnn.hu.poisson_sampler import PoissonSampler


class A2SPoissonCodingSignConvert(A2SHU):
    '''采样器为泊松采样器、精度转换函数为符号函数的A2SHU
    '''
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        super(A2SPoissonCodingSignConvert, self).__init__(
            window_size, torch.sign, non_linear)
        self.sampler = PoissonSampler(window_size=self.window_size)
        self.check()