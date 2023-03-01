# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.a2s_learnable_coding import A2SLearnableCoding


class A2SLearnableCodingSignConvert(A2SLearnableCoding):
    '''使用可学习采样器并且精度转换函数为符号函数的A2SHU
    '''
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        super(A2SLearnableCodingSignConvert, self).__init__(
            window_size, torch.sign, non_linear)