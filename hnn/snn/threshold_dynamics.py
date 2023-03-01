# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_model import QModel
from hnn.snn.leaky import Leaky, QLeaky
from hnn.snn.soft_update_after_spike import SoftUpdateAfterSpike, QSoftUpdateAfterSpike


class ThresholdDynamics(torch.nn.Module):
    '''膜电位阈值的动力学

    包括阈值自适应分量的指数衰减和发放后导致的阈值增加

    Args:
        decay.alpha = v_th_alpha
        decay.beta = v_th_beta
        decay.adpt_en = v_th_adpt_en
        update.value = v_th_incre
    '''
    def __init__(self, v_th_alpha, v_th_beta, v_th_incre, v_th_adpt_en=True) -> None:
        super(ThresholdDynamics, self).__init__()
        self.decay = Leaky(alpha=v_th_alpha, beta=v_th_beta,
                           adpt_en=v_th_adpt_en)
        self.update = SoftUpdateAfterSpike(value=v_th_incre)

    def forward(self, v_th_adpt: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        v_th_adpt = self.decay(v_th_adpt)
        v_th_adpt = self.update(v_th_adpt, spike)
        return v_th_adpt


class QThresholdDynamics(QModel):
    '''支持量化的膜电位阈值的动力学
    '''
    def __init__(self, v_th_alpha, v_th_beta, v_th_incre, v_th_adpt_en=True) -> None:
        QModel.__init__(self)
        self.decay = QLeaky(
            alpha=v_th_alpha, beta=v_th_beta, adpt_en=v_th_adpt_en)
        self.update = QSoftUpdateAfterSpike(value=v_th_incre)

    def forward(self, v_th_adpt: torch.Tensor, spike: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        v_th_adpt = self.decay.forward(x=v_th_adpt, weight_scale=scale)
        v_th_adpt = self.update.forward(
            x=v_th_adpt, spike=spike, weight_scale=scale)
        return v_th_adpt