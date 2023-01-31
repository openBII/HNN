# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import unittest
from src.snn.q_module import QModule
from src.snn.lif import LIF, QLIF
from src.grad import FakeQuantizeINT28


class RefQLIF(QModule, LIF):
    def __init__(self, v_th_0, v_leaky_alpha=1, v_leaky_beta=0, v_reset=0, v_leaky_adpt_en=False, v_init=None):
        QModule.__init__(self)
        LIF.__init__(self, v_th_0, v_leaky_alpha, v_leaky_beta,
                     v_reset, v_leaky_adpt_en, v_init)
        self.weight_scale = None
        self.first_time = True
        self.pretrained = False

    def collect_q_params(self):
        QModule.collect_q_params(self)

    def forward(self, x, weight_scale, v=None):
        self.weight_scale = weight_scale
        if self.quantization_mode:
            v = self._quantize(v)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            if v is not None:
                v = FakeQuantizeINT28.apply(v, weight_scale)
        spike, v = LIF.forward(self, x, v)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            v = v.clamp(-134217728, 134217727)  # INT28
        return spike, v

    def quantize(self):
        QModule.quantize(self)

    def _quantize(self, v: torch.Tensor) -> torch.Tensor:
        if self.first_time:
            self.first_time = False
            if not self.pretrained:
                self.if_node.fire.v_th = round(
                    self.if_node.fire.v_th * self.weight_scale.item())
                self.if_node.accumulate.v_init = round(
                    self.if_node.accumulate.v_init * self.weight_scale.item())
                self.if_node.reset.value = round(
                    self.if_node.reset.value * self.weight_scale.item())
                self.v_leaky.beta = round(
                    self.v_leaky.beta * self.weight_scale.item())
                if self.v_leaky.adpt_en:
                    self.v_leaky.alpha = round(self.v_leaky.alpha * 256) / 256
            if v is not None:
                v = v.mul(self.weight_scale).round(
                ).clamp(-134217728, 134217727)
        return v

    def dequantize(self):
        QModule.dequantize(self)
        self.if_node.fire.v_th = self.if_node.fire.v_th / self.weight_scale.item()
        self.if_node.reset.value = self.if_node.reset.value / self.weight_scale.item()
        self.v_leaky.beta = self.v_leaky.beta / self.weight_scale.item()
        self.if_node.accumulate.v_init = self.if_node.accumulate.v_init / \
            self.weight_scale.item()

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.if_node.fire.v_th = round(
            self.if_node.fire.v_th * self.weight_scale.item()) / self.weight_scale.item()
        self.if_node.reset.value = round(
            self.if_node.reset.value * self.weight_scale.item()) / self.weight_scale.item()
        self.if_node.accumulate.v_init = round(
            self.if_node.accumulate.v_init * self.weight_scale.item()) / self.weight_scale.item()
        self.v_leaky.beta = round(
            self.v_leaky.beta * self.weight_scale.item()) / self.weight_scale.item()
        if self.v_leaky.adpt_en:
            self.v_leaky.alpha = round(self.v_leaky.alpha * 256) / 256



class TestLIF(unittest.TestCase):
    def test_lif(self):
        x = torch.randn(1, 1, 28, 28)
        x = (x > 0).float()
        lif = RefQLIF(v_th_0=1, v_leaky_alpha=0.9, v_leaky_beta=0.5,
                      v_reset=0.2, v_leaky_adpt_en=True, v_init=0.1)
        new_lif = QLIF(v_th=1, v_leaky_alpha=0.9, v_leaky_beta=0.5,
                    v_reset=0.2, v_leaky_adpt_en=True, v_init=0.1)
        lif.quantize()
        new_lif.quantize()
        scale = torch.as_tensor(100)
        _, v_ref = lif.forward(x, scale)
        _, v = new_lif.forward(x, scale)
        self.assertTrue((v_ref - v).abs().mean() < 1)


if __name__ == '__main__':
    t1 = TestLIF()
    t1.test_lif()