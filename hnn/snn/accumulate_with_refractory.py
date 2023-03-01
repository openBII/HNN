# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_module import QModule
from hnn.grad import FakeQuantizeINT28


class AccumulateWithRefractory(torch.nn.Module):
    '''考虑不应期的膜电位累加

    不应期计数不为0时不进行膜电位累加

    Args:
        v_init: 如果输入膜电位为None, 则输入膜电位默认为固定初始值
    '''
    def __init__(self, v_init) -> None:
        super(AccumulateWithRefractory, self).__init__()
        self.v_init = v_init

    def forward(self, u_in: torch.Tensor, v=None, ref_cnt=None) -> torch.Tensor:
        if v is None:
            v = torch.full_like(u_in, self.v_init)
        if ref_cnt is None:
            ref_cnt = torch.zeros_like(u_in)
        ref_mask = (1 - ref_cnt).clamp(min=0)
        return u_in * ref_mask + v


class QAccumulateWithRefractory(QModule, AccumulateWithRefractory):
    '''支持量化的考虑不应期的膜电位累加操作

    其他说明类似于hnn/snn/q_accumulate.py
    '''
    def __init__(self, v_init) -> None:
        QModule.__init__(self)
        AccumulateWithRefractory.__init__(self, v_init)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False
        self.freeze = False

    def forward(self, u_in, weight_scale: torch.Tensor, v=None, ref_cnt=None):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            v = self._quantize(v)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            if v is not None:
                v = FakeQuantizeINT28.apply(v, weight_scale)
        v = AccumulateWithRefractory.forward(self, u_in, v, ref_cnt)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            v = v.clamp(-134217728, 134217727)  # INT28
        return v

    def _quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.first_time:
            self.first_time = False
            if not self.pretrained and not self.freeze:
                self.v_init = round(self.v_init * self.weight_scale.item())
            if v is not None:
                v = v.mul(self.weight_scale).round(
                ).clamp(-134217728, 134217727)
        return v

    def dequantize(self):
        QModule.dequantize(self)
        self.v_init = self.v_init / self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.v_init = round(
            self.v_init * self.weight_scale.item()) / self.weight_scale.item()