# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.grad import FakeQuantizeINT28
from hnn.snn.q_module import QModule


class ThresholdAccumulate(torch.nn.Module):
    '''膜电位阈值累加

    Args:
        v_th0: 固定阈值
    '''
    def __init__(self, v_th0) -> None:
        super(ThresholdAccumulate, self).__init__()
        self.v_th0 = v_th0

    def forward(self, v_th_adpt) -> torch.Tensor:
        with torch.no_grad():
            v_th_adpt = torch.as_tensor(v_th_adpt)
            v_th = self.v_th0 + v_th_adpt
        return v_th


class QThresholdAccumulate(QModule, ThresholdAccumulate):
    '''支持量化的膜电位阈值累加

    其他说明类似于hnn/snn/accumulate.py
    '''
    def __init__(self, v_th0):
        QModule.__init__(self)
        ThresholdAccumulate.__init__(self, v_th0)
        self.scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, v_th_adpt, scale: torch.Tensor) -> torch.Tensor:
        self.scale = scale
        if self.quantization_mode:
            self._quantize(v_th_adpt)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            v_th_adpt = FakeQuantizeINT28.apply(v_th_adpt, scale)
        # forward
        v_th = ThresholdAccumulate.forward(self, v_th_adpt)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            v_th = v_th.clamp(-134217728, 134217727)  # INT28
        return v_th

    def _quantize(self, v) -> torch.Tensor:
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.v_th0 = round(self.v_th0 * self.scale.item())
            v = torch.as_tensor(v)
            v = v.mul(self.scale).round().clamp(-134217728, 134217727)
        return v

    def dequantize(self):
        QModule.dequantize(self)
        self.v_th0 = self.v_th0 / self.scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.v_th0 = round(self.v_th0 * self.scale.item()) / self.scale.item()