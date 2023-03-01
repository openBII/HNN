# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.a2shu import A2SHU
from typing import Callable
from hnn.hu.learnable_sampler import LearnableSampler


class A2SLearnableCoding(A2SHU):
    '''使用可学习采样器的A2SHU
    '''
    def __init__(self, window_size: int, converter: Callable[[torch.Tensor], torch.Tensor],
                 non_linear: torch.nn.Module = None) -> None:
        super(A2SLearnableCoding, self).__init__(
            window_size, converter, non_linear)
        self.sampler = LearnableSampler(window_size=self.window_size)
        self.check()
