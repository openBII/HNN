# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch


class WindowSet(torch.nn.Module):
    '''设置时间窗

    非抽象类, 已给出具体实现
    假设输入数据维度为[batch_size, .., T], 最后一个维度为时间, 时间窗长度为t, 输出数据维度变为[batch_size, ..., T / t, t]

    Args:
        size: 时间窗长度
    '''
    def __init__(self, size: int) -> None:
        super(WindowSet, self).__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(-1)
        num = t // self.size
        shape = list(x.size())
        shape[-1] = num
        shape.append(self.size)
        x = x.unsqueeze(-2).reshape(shape)
        return x