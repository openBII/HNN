# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_module import QModule


class FireWithConstantThreshold(torch.nn.Module):
    '''固定阈值的脉冲发放

    一层神经元共享相同的阈值

    Args:
        surrogate_function: 梯度替代函数, 可使用的函数见hnn/snn/surrogate
        v_th: 固定的阈值常量
    '''
    def __init__(self, surrogate_function, v_th) -> None:
        super(FireWithConstantThreshold, self).__init__()
        self.surrogate_function = surrogate_function
        self.v_th = v_th

    def forward(self, v) -> torch.Tensor:
        spike = self.surrogate_function.apply(v, self.v_th)
        return spike


class QFireWithConstantThreshold(QModule, FireWithConstantThreshold):
    '''支持量化的固定阈值的脉冲发放

    其他说明类似于hnn/snn/accumulate.py
    '''
    def __init__(self, surrogate_function, v_th) -> None:
        QModule.__init__(self)
        FireWithConstantThreshold.__init__(self, surrogate_function, v_th)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, v: torch.Tensor, weight_scale: torch.Tensor):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            self._quantize()
        spike = FireWithConstantThreshold.forward(self, v)
        return spike

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.v_th = round(self.v_th * self.weight_scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        self.v_th = self.v_th / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.v_th = round(self.v_th * self.weight_scale.item()
                          ) / self.weight_scale.item()