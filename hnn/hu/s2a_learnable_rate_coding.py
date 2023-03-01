# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.s2ahu import S2AHU
from hnn.hu.learnable_window_conv import LearnableWindowConv
from typing import Union, Tuple


class S2ALearnableRateCoding(S2AHU):
    '''使用可学习时间窗卷积的S2AHU

    Args:
        window_size: 时间窗大小
        num_windows: 时间窗数量
        kernel_size: 卷积窗大小
        stride: 卷积步长
        padding: 卷积补零
        non_linear: 非线性函数
    '''
    def __init__(self, window_size: int, num_windows: int, kernel_size: int,
                 stride: int, padding: Union[int, Tuple[int]], non_linear: torch.nn.Module = None) -> None:
        super(S2ALearnableRateCoding, self).__init__(window_size, non_linear)
        self.window_conv = LearnableWindowConv(
            in_channels=num_windows, kernel_size=kernel_size, stride=stride, padding=padding)
        self.check()