# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch


class Fire(torch.nn.Module):
    '''脉冲发放

    膜电位和阈值均为张量, 即每个神经元都可以有不同的阈值

    Args:
        surrogate_function: 梯度替代函数, 可使用的函数见hnn/snn/surrogate
    '''
    def __init__(self, surrogate_function) -> None:
        super(Fire, self).__init__()
        self.surrogate_function = surrogate_function

    def forward(self, v, v_th) -> torch.Tensor:
        spike = self.surrogate_function.apply(v, v_th)
        return spike