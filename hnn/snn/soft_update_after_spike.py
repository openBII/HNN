# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_module import QModule


class SoftUpdateAfterSpike(torch.nn.Module):
    '''发放脉冲后对输入进行soft reset

    soft reset包含两种模式, constant模式在输入上加上一个常量, 否则对输入加一个张量

    Args:
        value: constant模式下的常量, default = None
    '''
    def __init__(self, value=None) -> None:
        super(SoftUpdateAfterSpike, self).__init__()
        self.value = value

    def forward(self, x: torch.Tensor, spike: torch.Tensor, update: torch.Tensor = None):
        if self.value is None:
            assert update is not None
            out = x + spike * update
        else:
            out = x + spike * self.value
        return out


class QSoftUpdateAfterSpike(QModule, SoftUpdateAfterSpike):
    '''支持量化的发放脉冲后对输入进行soft reset操作

    其他说明类似于hnn/snn/accumulate.py
    '''
    def __init__(self, value=None) -> None:
        QModule.__init__(self)
        SoftUpdateAfterSpike.__init__(self, value)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, x: torch.Tensor, spike: torch.Tensor, weight_scale: torch.Tensor, update: torch.Tensor = None):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            self._quantize()
        x = SoftUpdateAfterSpike.forward(self, x, spike, update)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = x.clamp(-134217728, 134217727)  # INT28
        return x

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                if self.value is not None:
                    self.value = round(self.value * self.weight_scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        if self.value is not None:
            self.value = self.value / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        if self.value is not None:
            self.value = round(
                self.value * self.weight_scale.item()) / self.weight_scale.item()