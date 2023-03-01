# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.average_window_conv import AverageWindowConv
from typing import List


class GlobalAverageWindowConv(AverageWindowConv):
    '''全局平均时间窗卷积

    平均时间窗卷积的退化情况, 在时间窗内计算平均值

    Args:
        window_size: 时间窗大小
    '''
    def __init__(self, window_size: int) -> None:
        super(GlobalAverageWindowConv, self).__init__(
            kernel_size=window_size, stride=window_size)

    def reshape(self, x: torch.Tensor, prefix_shape: List) -> torch.Tensor:
        '''重载的卷积后reshape方法

        和父类的reshape方法效果相同, 但更简洁
        '''
        prefix_shape.append(x.size(-2))
        x = x.squeeze(-1).reshape(prefix_shape)
        return x