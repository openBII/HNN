# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_dynamics import QDynamics
from hnn.grad import FakeQuantizeINT28


class Accumulate(torch.nn.Module):
    '''膜电位累加

    Args:
        v_init: 如果输入膜电位为None, 则输入膜电位默认为固定初始值
    '''
    def __init__(self, v_init) -> None:
        super(Accumulate, self).__init__()
        self.v_init = v_init

    def forward(self, u_in, v=None) -> torch.Tensor:
        if v is None:
            v = torch.full_like(u_in, self.v_init)
        return u_in + v


class QAccumulate(QDynamics, Accumulate):
    '''支持量化的膜电位累加操作
    '''
    def __init__(self, v_init) -> None:
        QDynamics.__init__(self)
        Accumulate.__init__(self, v_init)

    def forward(self, x, scale: torch.Tensor, v=None):
        self.scale = scale
        if self.quantization_mode:
            v = self._quantize(v)
        if self.aware_mode:
            v = self._aware(v)
        v = Accumulate.forward(self, x, v)
        if self.quantization_mode:
            v = v.clamp(-134217728, 134217727)  # INT28
        return v

    def _quantize(self, v: torch.Tensor) -> torch.Tensor:
        '''运行时量化方法

        由于SNN脉冲神经元的量化参数需要Integrate阶段的算子给出, 所以此方法在运行时被调用
        只有初次被调用会对输入膜电位进行量化
        两次推理过程中调用refresh方法可以冻结神经元参数, 只有没有加载预训练模型且不处于冻结状态时会对脉冲神经元参数进行量化
        '''
        if self.first_time:
            self.first_time = False
            if not self.freeze:
                self.v_init = round(self.v_init * self.scale.item())
            if v is not None:
                v = v.mul(self.scale).round(
                ).clamp(-134217728, 134217727)
        return v

    def dequantize(self):
        QDynamics.dequantize(self)
        self.v_init = self.v_init / self.scale.item()

    def _aware(self, v: torch.Tensor):
        if self.quantization_mode:
            self.dequantize()
        self.v_init = round(
            self.v_init * self.scale.item()) / self.scale.item()
        if v is not None:
            v = FakeQuantizeINT28.apply(v, self.scale)
        return v