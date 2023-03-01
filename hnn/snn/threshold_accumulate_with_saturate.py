# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_model import QModel
from hnn.snn.threshold_accumulate import ThresholdAccumulate, QThresholdAccumulate
from hnn.snn.saturate import Saturate, QSaturate


class ThresholdAccumulateWithSaturate(torch.nn.Module):
    '''带有下限饱和的膜电位阈值累加

    包括膜电位阈值累加和下限饱和两个步骤

    Args:
        accumulate.vth0 = v_th0
        saturate.v_l = v_l
    '''
    def __init__(self, v_th0, v_l) -> None:
        super(ThresholdAccumulateWithSaturate, self).__init__()
        self.accumulate = ThresholdAccumulate(v_th0=v_th0)
        self.saturate = Saturate(v_l=v_l)

    def forward(self, v_th_adpt) -> torch.Tensor:
        with torch.no_grad():
            v_th = self.accumulate(v_th_adpt)
            v_th = self.saturate(v_th)
        return v_th


class QThresholdAccumulateWithSaturate(QModel):
    '''支持量化的带有下限饱和的膜电位阈值累加
    '''
    def __init__(self, v_th0, v_l) -> None:
        QModel.__init__(self)
        self.accumulate = QThresholdAccumulate(v_th0=v_th0)
        self.saturate = QSaturate(v_l=v_l)

    def forward(self, v_th_adpt, scale) -> torch.Tensor:
        with torch.no_grad():
            v_th = self.accumulate.forward(v_th_adpt=v_th_adpt, scale=scale)
            v_th = self.saturate.forward(x=v_th_adpt, scale=scale)
        return v_th