# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.rate_coding_sampler import RateCodingSampler
from spikingjelly.clock_driven import encoding


class PoissonSampler(RateCodingSampler):
    '''泊松采样器

    首先通过正则化将数据转换到0到1区间, 然后通过泊松采样器采样, 这里复用了SpikingJelly中的PoissonEncoder
    '''
    def __init__(self, window_size: int) -> None:
        super().__init__(window_size, encoding.PoissonEncoder())

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.min()) / (x.max() - x.min())