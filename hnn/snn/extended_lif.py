# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_model import QModel
from hnn.snn.reset_mode import ResetMode
from hnn.snn.refractory import Refractory
from hnn.snn.threshold_dynamics import ThresholdDynamics, QThresholdDynamics
from hnn.snn.saturate import Saturate, QSaturate
from hnn.snn.threshold_accumulate_with_saturate import ThresholdAccumulateWithSaturate, QThresholdAccumulateWithSaturate
from hnn.snn.lif_with_tensor_threshold_and_reset_mode_and_refractory import LIFWithTensorThresholdAndResetModeAndRefractory, QLIFWithTensorThresholdAndResetModeAndRefractory


class ExtendedLIF(torch.nn.Module):
    '''扩展LIF神经元

    包括以下阶段:
    1. 膜电位阈值累加, 支持下限饱和
    2. LIF神经元计算, 支持不同神经元可以有不同的阈值, 支持可配置的膜电位复位模式和不应期
    3. 膜电位下限饱和
    4. 膜电位阈值动力学, 包括阈值自适应分量的指数衰减和发放后导致的阈值增加
    5. 不应期减计数, 发放脉冲后不应期复位

    Integrate阶段通过其他算子完成

    Args:
        threshold_accumulate.accumulate.vth0 = v_th0
        threshold_accumulate.saturate.v_l = v_l
        lif.if_node.reset.reset_mode = reset_mode
        lif.if_node.reset.v_reset = v_reset
        lif.if_node.reset.dv = dv
        lif.if_node.accumulate.v_init = v_init
        lif.if_node.fire.surrogate_function: 默认为Rectangle
        window_size: Rectangle的矩形窗宽度, default = 1
        lif.v_leaky.alpha = v_leaky_alpha
        lif.v_leaky.beta = v_leaky_beta
        lif.v_leaky.adpt_en = v_leaky_adpt_en
        saturate.v_l = v_l
        refractory.reset.value = ref_len
        threshold_dynamics.decay.alpha = v_th_alpha
        threshold_dynamics.decay.beta = v_th_beta
        threshold_dynamics.decay.adpt_en = v_th_adpt_en
        threshold_dynamics.update.value = v_th_incre
    '''
    def __init__(self, v_th0,
                 v_leaky_alpha=1, v_leaky_beta=0,
                 v_leaky_adpt_en=False,
                 v_reset=0, v_init=None,
                 v_th_alpha=1, v_th_beta=0, v_th_adpt_en=True,
                 v_th_incre=0, v_l=None, dv=0,
                 ref_len=0, reset_mode=ResetMode.HARD,
                 window_size=1):
        self.threshold_accumulate = ThresholdAccumulateWithSaturate(
            v_th0=v_th0, v_l=v_l)
        self.lif = LIFWithTensorThresholdAndResetModeAndRefractory(
            v_leaky_alpha=v_leaky_alpha, v_leaky_beta=v_leaky_beta, reset_mode=reset_mode,
            v_reset=v_reset, dv=dv, v_leaky_adpt_en=v_leaky_adpt_en, v_init=v_init, window_size=window_size)
        self.saturate = Saturate(v_l=v_l)
        self.refractory = Refractory(ref_len=ref_len)
        self.threshold_dynamics = ThresholdDynamics(
            v_th_alpha=v_th_alpha, v_th_beta=v_th_beta, v_th_incre=v_th_incre, v_th_adpt_en=v_th_adpt_en)

    def forward(self, u_in, v_th_adpt, v=None, ref_cnt=None):
        v_th = self.threshold_accumulate.forward(v_th_adpt)
        spike, v = self.lif.forward(u_in, v_th, v, ref_cnt)
        v = self.saturate.forward(v)
        v_th_adpt = self.threshold_dynamics.forward(v_th_adpt, spike)
        ref_cnt = self.refractory.forward(ref_cnt, spike)
        return spike, v, v_th_adpt, ref_cnt


class QExtendedLIF(QModel):
    '''支持量化的扩展LIF神经元
    '''
    def __init__(self, v_th0,
                 v_leaky_alpha=1, v_leaky_beta=0,
                 v_reset=0, v_init=None,
                 v_th_alpha=1, v_th_beta=0,
                 v_th_incre=0, v_l=None, dv=0,
                 ref_len=0, reset_mode=ResetMode.HARD,
                 window_size=1):
        QModel.__init__(self)
        self.threshold_accumulate = QThresholdAccumulateWithSaturate(
            v_th0=v_th0, v_l=v_l)
        self.lif = QLIFWithTensorThresholdAndResetModeAndRefractory(
            v_leaky_alpha=v_leaky_alpha, v_leaky_beta=v_leaky_beta, reset_mode=reset_mode,
            v_reset=v_reset, dv=dv, v_init=v_init, window_size=window_size)
        self.saturate = QSaturate(v_l=v_l)
        self.refractory = Refractory(ref_len=ref_len)
        self.threshold_dynamics = QThresholdDynamics(
            v_th_alpha=v_th_alpha, v_th_beta=v_th_beta, v_th_incre=v_th_incre)

    def forward(self, u_in, v_th_adpt, scale, v=None, ref_cnt=None):
        v_th = self.threshold_accumulate.forward(v_th_adpt, scale)
        spike, v = self.lif.forward(u_in, v_th, scale, v, ref_cnt)
        v = self.saturate.forward(v, scale)
        v_th_adpt = self.threshold_dynamics.forward(v_th_adpt, spike, scale)
        ref_cnt = self.refractory.forward(ref_cnt, spike)
        return spike, v, v_th_adpt, ref_cnt