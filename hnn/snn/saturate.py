# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.grad import FakeQuantizeINT28
from hnn.snn.q_module import QModule


class Saturate(torch.nn.Module):
    '''下限饱和

    当输入低于阈值时, 输入取阈值

    Args:
        v_l: 下限饱和阈值
    '''
    def __init__(self, v_l):
        super(Saturate, self).__init__()
        self.v_l = v_l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clamp(min=self.v_l)
        return out


class QSaturate(QModule, Saturate):
    '''支持量化的下限饱和

    其他说明类似于hnn/snn/accumulate.py
    '''
    def __init__(self, v_l):
        QModule.__init__(self)
        Saturate.__init__(self, v_l)
        self.scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        self.scale = scale
        if self.quantization_mode:
            self._quantize()
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = FakeQuantizeINT28.apply(x, scale)
        # forward
        x = Saturate.forward(self, x)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = x.clamp(-134217728, 134217727)  # INT28
        return x

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.v_l = round(self.v_l * self.scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        self.v_l = self.v_l / self.scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.v_l = round(self.v_l * self.scale.item()) / self.scale.item()