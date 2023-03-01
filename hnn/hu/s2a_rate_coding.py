# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.s2ahu import S2AHU
from hnn.hu.average_window_conv import AverageWindowConv


class S2ARateCoding(S2AHU):
    '''使用平均时间窗卷积的S2AHU

    Args:
        window_size: 时间窗大小
        kernel_size: 卷积窗大小
        stride: 卷积窗滑动步长
        non_linear: 非线性函数
    '''
    def __init__(self, window_size: int, kernel_size: int, stride: int, non_linear: torch.nn.Module = None) -> None:
        super(S2ARateCoding, self).__init__(window_size, non_linear)
        self.window_conv = AverageWindowConv(
            kernel_size=kernel_size, stride=stride)
        self.check()