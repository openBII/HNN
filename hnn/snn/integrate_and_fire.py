# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_model import QModel
from hnn.snn.hard_update_after_spike import HardUpdateAfterSpike, QHardUpdateAfterSpike
from hnn.snn.surrogate.rectangle import Rectangle
from hnn.snn.accumulate import Accumulate, QAccumulate
from hnn.snn.fire_with_constant_threshold import FireWithConstantThreshold, QFireWithConstantThreshold


class IF(torch.nn.Module):
    '''Integrate-and-Fire神经元

    Integrate阶段通过其他算子完成
    包括膜电位累加、脉冲发放和膜电位复位三个阶段

    Args:
        reset.value = v_reset
        accumulate.v_init = v_init
        fire.v_th = v_th
        fire.surrogate_function: 默认为Rectangle
        window_size: Rectangle的矩形窗宽度, default = 1
    '''
    def __init__(self, v_th, v_reset, v_init=None, window_size=1):
        super(IF, self).__init__()
        self.reset = HardUpdateAfterSpike(value=v_reset)
        self.accumulate = Accumulate(
            v_init=self.reset.value if v_init is None else v_init)
        Rectangle.window_size = window_size
        self.fire = FireWithConstantThreshold(
            surrogate_function=Rectangle, v_th=v_th)

    def forward(self, u_in: torch.Tensor, v: torch.Tensor = None):
        # update
        v_update = self.accumulate(u_in, v)
        # fire
        spike = self.fire(v_update)
        v = self.reset(v_update, spike)
        return spike, v


class QIF(QModel):
    '''支持量化的IF神经元
    '''
    def __init__(self, v_th, v_reset, v_init=None, window_size=1):
        QModel.__init__(self)
        self.reset = QHardUpdateAfterSpike(value=v_reset)
        self.accumulate = QAccumulate(
            v_init=self.reset.value if v_init is None else v_init)
        Rectangle.window_size = window_size
        self.fire = QFireWithConstantThreshold(
            surrogate_function=Rectangle, v_th=v_th)

    def forward(self, u_in: torch.Tensor, scale: torch.Tensor, v: torch.Tensor = None):
        # update
        v_update = self.accumulate.forward(u_in, scale, v)
        # fire
        spike = self.fire.forward(v_update, scale)
        v = self.reset.forward(v_update, spike, scale)
        return spike, v