# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_model import QModel
from hnn.snn.reset_mode import ResetMode
from hnn.snn.hard_update_after_spike import HardUpdateAfterSpike, QHardUpdateAfterSpike
from hnn.snn.soft_update_after_spike import SoftUpdateAfterSpike, QSoftUpdateAfterSpike


class ResetAfterSpike(torch.nn.Module):
    '''脉冲发放后膜电位复位

    根据reset_mode不同选择不同的发放模式

    Args:
        reset_mode: 发放模式
        reset.v_reset: 当发放模式为HARD时的复位值
        reset.dv: 当发放模式为SOFT_CONSTANT时膜电位减去dv
    '''
    def __init__(self, reset_mode: ResetMode, v_reset=None, dv=None) -> None:
        super(ResetAfterSpike, self).__init__()
        self.reset_mode = reset_mode
        if self.reset_mode == ResetMode.HARD:
            assert v_reset is not None
            self.reset = HardUpdateAfterSpike(value=v_reset)
        elif self.reset_mode == ResetMode.SOFT_CONSTANT:
            assert dv is not None
            self.reset = SoftUpdateAfterSpike(value=-dv)
        else:
            assert self.reset_mode == ResetMode.SOFT, "Invalid reset mode"
            self.reset = SoftUpdateAfterSpike()

    def forward(self, v: torch.Tensor, spike: torch.Tensor, update: torch.Tensor = None):
        if self.reset_mode == ResetMode.SOFT:
            out = self.reset(v, spike, -update)
        else:
            out = self.reset(v, spike)
        return out


class QResetAfterSpike(QModel):
    '''支持量化的脉冲发放后膜电位复位操作
    '''
    def __init__(self, reset_mode: ResetMode, v_reset=None, dv=None) -> None:
        QModel.__init__(self)
        self.reset_mode = reset_mode
        if self.reset_mode == ResetMode.HARD:
            assert v_reset is not None
            self.reset = QHardUpdateAfterSpike(value=v_reset)
        elif self.reset_mode == ResetMode.SOFT_CONSTANT:
            assert dv is not None
            self.reset = QSoftUpdateAfterSpike(value=-dv)
        else:
            assert self.reset_mode == ResetMode.SOFT, "Invalid reset mode"
            self.reset = SoftUpdateAfterSpike()

    def forward(self, v: torch.Tensor, spike: torch.Tensor, scale: torch.Tensor, update: torch.Tensor = None):
        if self.reset_mode == ResetMode.SOFT:
            out = self.reset.forward(v, spike, -update)  # 这里要求update必须已经被量化过
        else:
            out = self.reset.forward(v, spike, weight_scale=scale)
        return out