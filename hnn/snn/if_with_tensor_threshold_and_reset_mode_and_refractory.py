# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_model import QModel
from hnn.snn.reset_mode import ResetMode
from hnn.snn.reset_after_spike import ResetAfterSpike, QResetAfterSpike
from hnn.snn.surrogate.rectangle import Rectangle
from hnn.snn.accumulate_with_refractory import AccumulateWithRefractory, QAccumulateWithRefractory
from hnn.snn.fire import Fire


class IFWithTensorThresholdAndResetModeAndRefractory(torch.nn.Module):
    '''Integrate-and-Fire神经元, 不同神经元可以有不同的阈值, 支持可配置的膜电位复位模式和不应期

    Integrate阶段通过其他算子完成
    包括支持不应期的膜电位累加、支持不同神经元有不同阈值的脉冲发放和支持多种模式的膜电位复位三个阶段

    Args:
        reset.reset_mode = reset_mode
        reset.v_reset = v_reset
        reset.dv = dv
        accumulate.v_init = v_init
        fire.surrogate_function: 默认为Rectangle
        window_size: Rectangle的矩形窗宽度, default = 1
    '''
    def __init__(self, reset_mode: ResetMode, v_reset=None, dv=None, v_init=None, window_size=1):
        super(IFWithTensorThresholdAndResetModeAndRefractory, self).__init__()
        self.reset = ResetAfterSpike(
            reset_mode=reset_mode, v_reset=v_reset, dv=dv)
        self.accumulate = AccumulateWithRefractory(
            v_init=v_reset if v_init is None else v_init)
        Rectangle.window_size = window_size
        self.fire = Fire(surrogate_function=Rectangle)

    def forward(self, u_in: torch.Tensor, v_th: torch.Tensor, v: torch.Tensor = None, ref_cnt: torch.Tensor = None):
        # update
        v_update = self.accumulate(u_in, v, ref_cnt)
        # fire
        spike = self.fire(v_update, v_th)
        v = self.reset(v_update, spike)
        return spike, v


class QIFWithTensorThresholdAndResetModeAndRefractory(QModel):
    def __init__(self, reset_mode: ResetMode, v_reset=None, dv=None, v_init=None, window_size=1):
        QModel.__init__(self)
        self.reset = QResetAfterSpike(
            reset_mode=reset_mode, v_reset=v_reset, dv=dv)
        self.accumulate = QAccumulateWithRefractory(
            v_init=v_reset if v_init is None else v_init)
        Rectangle.window_size = window_size
        self.fire = Fire(surrogate_function=Rectangle)

    def forward(self, u_in: torch.Tensor, v_th: torch.Tensor, scale: torch.Tensor, v: torch.Tensor = None, ref_cnt: torch.Tensor = None):
        # update
        v_update = self.accumulate.forward(u_in, scale, v, ref_cnt)
        # fire
        spike = self.fire.forward(v_update, v_th)
        v = self.reset.forward(v_update, spike, scale, v_th)
        return spike, v