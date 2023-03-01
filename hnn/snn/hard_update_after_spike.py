# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_module import QModule


class HardUpdateAfterSpike(torch.nn.Module):
    '''发放脉冲后对输入进行hard reset

    Attributes:
        value: 重置后的值
    '''
    def __init__(self, value: float) -> None:
        super(HardUpdateAfterSpike, self).__init__()
        self.value = value

    def forward(self, x: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        out = spike * self.value + (1 - spike) * x  # 避免inplace操作同时保证可导
        return out


class QHardUpdateAfterSpike(QModule, HardUpdateAfterSpike):
    '''支持量化的发放脉冲后对输入进行hard reset操作

    其他说明类似于hnn/snn/q_accumulate.py
    '''
    def __init__(self, value) -> None:
        QModule.__init__(self)
        HardUpdateAfterSpike.__init__(self, value)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, x: torch.Tensor, spike: torch.Tensor, weight_scale: torch.Tensor):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            self._quantize()
        x = HardUpdateAfterSpike.forward(self, x, spike)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = x.clamp(-134217728, 134217727)  # INT28
        return x

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.value = round(self.value * self.weight_scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        self.value = self.value / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.value = round(self.value * self.weight_scale.item()
                           ) / self.weight_scale.item()
