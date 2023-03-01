# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_model import QModel
from hnn.snn.if_with_tensor_threshold_and_reset_mode_and_refractory import IFWithTensorThresholdAndResetModeAndRefractory, QIFWithTensorThresholdAndResetModeAndRefractory
from hnn.snn.reset_mode import ResetMode
from hnn.snn.leaky import Leaky, QLeaky


class LIFWithTensorThresholdAndResetModeAndRefractory(torch.nn.Module):
    '''Leaky-Integrate-and-Fire神经元, 不同神经元可以有不同的阈值, 支持可配置的膜电位复位模式和不应期

    Integrate阶段通过其他算子完成
    包括支持不应期的膜电位累加、支持不同神经元有不同阈值的脉冲发放、支持多种模式的膜电位复位和膜电位泄漏四个阶段

    Args:
        if_node.reset.reset_mode = reset_mode
        if_node.reset.v_reset = v_reset
        if_node.reset.dv = dv
        if_node.accumulate.v_init = v_init
        if_node.fire.surrogate_function: 默认为Rectangle
        window_size: Rectangle的矩形窗宽度, default = 1
        v_leaky.alpha = v_leaky_alpha
        v_leaky.beta = v_leaky_beta
        v_leaky.adpt_en = v_leaky_adpt_en
    '''
    def __init__(self, v_leaky_alpha, v_leaky_beta, reset_mode: ResetMode, v_reset=None, dv=None, v_leaky_adpt_en=False, v_init=None, window_size=1):
        super(LIFWithTensorThresholdAndResetModeAndRefractory, self).__init__()
        self.if_node = IFWithTensorThresholdAndResetModeAndRefractory(
            reset_mode=reset_mode, v_reset=v_reset, dv=dv, v_init=v_init, window_size=window_size)
        self.v_leaky = Leaky(alpha=v_leaky_alpha,
                             beta=v_leaky_beta, adpt_en=v_leaky_adpt_en)

    def forward(self, u_in: torch.Tensor, v_th, v=None, ref_cnt=None):
        spike, v = self.if_node(u_in, v_th, v, ref_cnt)
        v = self.v_leaky(v)
        return spike, v


class QLIFWithTensorThresholdAndResetModeAndRefractory(QModel):
    def __init__(self, v_leaky_alpha, v_leaky_beta, reset_mode: ResetMode, v_reset=None, dv=None, v_leaky_adpt_en=False, v_init=None, window_size=1):
        QModel.__init__(self)
        self.if_node = QIFWithTensorThresholdAndResetModeAndRefractory(
            reset_mode=reset_mode, v_reset=v_reset, dv=dv, v_init=v_init, window_size=window_size)
        self.v_leaky = QLeaky(alpha=v_leaky_alpha,
                              beta=v_leaky_beta, adpt_en=v_leaky_adpt_en)

    def forward(self, u_in: torch.Tensor, v_th, scale, v=None, ref_cnt=None):
        spike, v = self.if_node.forward(u_in, v_th, scale, v, ref_cnt)
        v = self.v_leaky.forward(v, scale)
        return spike, v