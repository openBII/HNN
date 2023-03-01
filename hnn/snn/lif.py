# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_model import QModel
from hnn.snn.integrate_and_fire import IF, QIF
from hnn.snn.leaky import Leaky, QLeaky


class LIF(torch.nn.Module):
    '''Leaky-Integrate-and-Fire神经元

    Integrate阶段通过其他算子完成
    由IF神经元和Leaky操作组成

    Args:
        if_node.reset.value = v_reset
        if_node.accumulate.v_init = v_init
        if_node.fire.v_th = v_th
        if_node.fire.surrogate_function: 默认为Rectangle
        v_leaky.alpha = v_leaky_alpha
        v_leaky.beta = v_leaky_beta
        v_leaky.adpt_en = v_leaky_adpt_en
        window_size: Rectangle的矩形窗宽度, default = 1
    '''
    def __init__(self, v_th, v_leaky_alpha, v_leaky_beta, v_reset=0, v_leaky_adpt_en=False, v_init=None, window_size=1):
        super(LIF, self).__init__()
        self.if_node = IF(v_th=v_th, v_reset=v_reset,
                          v_init=v_init, window_size=window_size)
        self.v_leaky = Leaky(alpha=v_leaky_alpha,
                             beta=v_leaky_beta, adpt_en=v_leaky_adpt_en)

    def forward(self, u_in: torch.Tensor, v=None):
        spike, v = self.if_node(u_in, v)
        v = self.v_leaky(v)
        return spike, v


class QLIF(QModel):
    '''支持量化的LIF神经元
    '''
    def __init__(self, v_th, v_leaky_alpha, v_leaky_beta, v_reset=0, v_leaky_adpt_en=False, v_init=None, window_size=1):
        QModel.__init__(self)
        self.if_node = QIF(v_th=v_th, v_reset=v_reset,
                           v_init=v_init, window_size=window_size)
        self.v_leaky = QLeaky(alpha=v_leaky_alpha,
                              beta=v_leaky_beta, adpt_en=v_leaky_adpt_en)

    def forward(self, u_in: torch.Tensor, scale: torch.Tensor, v=None):
        spike, v = self.if_node.forward(u_in, scale, v)
        v = self.v_leaky.forward(v, scale)
        return spike, v