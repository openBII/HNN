# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from typing import List


class WindowConv(torch.nn.Module):
    '''时间窗内进行时间维度上的卷积

    抽象类
    '''
    def __init__(self) -> None:
        super(WindowConv, self).__init__()

    def reshape1d(self, x: torch.Tensor):
        '''将形状为[batch_size, ..., num_of_windows, window_size]的数据转换成[batch_size * ..., num_of_windows, window_size]

        此方法应用在进行卷积之前, 由于要进行1D卷积, 所以需要先将输入数据变成三维张量

        Returns:
            第一个返回值: reshape之后的三维数据
            第二个返回值: 原始数据的部分维度
        '''
        shape = list(x.size())
        batch_size = 1
        for i in range(0, len(shape) - 2):
            batch_size *= shape[i]
        num = x.size(-2)
        size = x.size(-1)
        x = x.reshape(batch_size, num, size)
        return x, shape[:-2]

    def reshape(self, x: torch.Tensor, prefix_shape: List) -> torch.Tensor:
        '''用于在完成时间窗卷积之后的形状变换

        假设输入特征图形状为[N, num_of_windows, window_size], 此方法将数据形状转换成[*prefix_shape, num_of_windows * window_size], 最后一个维度为时间维度

        Args:
            prefix_shape: reshape1d的第二个返回值, prod(prefix_shape) = N
        '''
        num = x.size(-2)
        size = x.size(-1)
        t = num * size
        prefix_shape.append(t)
        x = x.reshape(prefix_shape)
        return x