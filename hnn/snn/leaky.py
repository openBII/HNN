# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.grad import DifferentiableFloor
from hnn.snn.q_module import QModule


class Leaky(torch.nn.Module):
    '''指数衰减操作

    y = a x + b (a <= 1)

    Args:
        alpha: 指数衰减系数
        beta: 指数衰减常数
        adpt_en: 是否进行指数衰减, 否相当于alpha = 1
    '''
    def __init__(self, alpha, beta, adpt_en=True):
        super(Leaky, self).__init__()
        self.alpha = alpha
        self.beta = beta
        assert alpha <= 1
        self.adpt_en = adpt_en

    def forward(self, x: torch.Tensor):
        if self.adpt_en:
            out = self.alpha * x + self.beta
        else:
            out = x + self.beta
        return out


class QLeaky(QModule, Leaky):
    '''支持量化的指数衰减操作
    '''
    def __init__(self, alpha, beta, adpt_en=True):
        QModule.__init__(self)
        Leaky.__init__(self, alpha=alpha, beta=beta, adpt_en=adpt_en)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, x: torch.Tensor, weight_scale: torch.Tensor):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            self._quantize()
        # forward
        if self.adpt_en:
            if self.quantization_mode:
                x = torch.floor(self.alpha * x) + self.beta
            elif self.aware_mode:
                assert not(
                    self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
                x = DifferentiableFloor(
                    self.alpha * x * weight_scale) / weight_scale + self.beta
            else:
                x = self.alpha * x + self.beta
        else:
            x = x + self.beta
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = x.clamp(-134217728, 134217727)  # INT28
        return x

    def _quantize(self):
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.alpha = round(self.alpha * 256) / 256
                self.beta = round(self.beta * self.weight_scale.item())

    def dequantize(self):
        QModule.dequantize(self)
        self.beta = self.beta / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.alpha = round(self.alpha * 256) / 256
        self.beta = round(self.beta * self.weight_scale.item()
                          ) / self.weight_scale.item()