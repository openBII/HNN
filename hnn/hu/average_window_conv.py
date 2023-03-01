# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.window_conv import WindowConv


class AverageWindowConv(WindowConv):
    '''平均时间窗卷积

    时间窗卷积的退化情况, 在卷积窗内计算平均值

    Args:
        kernel_size: 卷积窗大小
        stride: 滑窗的步长
    '''
    def __init__(self, kernel_size: int, stride: int) -> None:
        super(AverageWindowConv, self).__init__()
        self.avgpool = torch.nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入特征图形状 [N, ..., num, size]
        # 输出特征图形状 [N, ..., T]
        x, prefix_shape = self.reshape1d(x)  # [N * ..., num, size], [N, ...]
        x = self.avgpool(x)
        x = self.reshape(x, prefix_shape)
        return x