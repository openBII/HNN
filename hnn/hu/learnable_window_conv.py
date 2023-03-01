# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.window_conv import WindowConv
from typing import Union, Tuple


class LearnableWindowConv(WindowConv):
    '''可学习的时间窗卷积

    进行groups = in_channels = out_channels的1D卷积, 类似于1D的在时间维度上的深度可分离卷积

    Args:
        in_channels: 卷积的输入通道数, 等于时间窗的数量
        kernel_size: 卷积核大小
        stride: 卷积的步长
        padding: 卷积的补零
    '''
    def __init__(self, in_channels: int, kernel_size: int, stride: int, padding: Union[int, Tuple[int]]) -> None:
        super(LearnableWindowConv, self).__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, prefix_shape = self.reshape1d(x)
        x = self.conv(x)
        x = self.reshape(x, prefix_shape)
        return x