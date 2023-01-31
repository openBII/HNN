# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch


class HU(torch.nn.Module):
    '''Hybrid Unit类

    Hybrid Unit抽象类, 包含五个部分:
    - window_set: 设置时间窗
    - window_conv: 时间窗内进行时间维度上的卷积
    - sampler: 采样器
    - non_linear: 非线性函数
    - precision_convert: 精度转换单元

    所有Hybrid Unit实例均继承于HU类, 不需要实现forward函数
    '''
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        '''HU构造函数

        Args:
            window_size: 时间窗大小
            non_linear: 非线性变换
        '''
        super(HU, self).__init__()
        self.window_size = window_size
        self.window_set = None
        self.window_conv = None
        self.sampler = None
        self.non_linear = non_linear
        self.precision_convert = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_set is not None:
            x = self.window_set(x)
        if self.window_conv is not None:
            x = self.window_conv(x)
        if self.sampler is not None:
            x = self.sampler(x)
        if self.non_linear is not None:
            x = self.non_linear(x)
        if self.precision_convert is not None:
            x = self.precision_convert(x)
        return x