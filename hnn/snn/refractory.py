# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.hard_update_after_spike import HardUpdateAfterSpike


class Refractory(torch.nn.Module):
    '''不应期

    不应期减计数, 发放脉冲后不应期复位

    Args:
        reset.value = ref_len, 不应期长度
    '''
    def __init__(self, ref_len) -> None:
        super(Refractory, self).__init__()
        self.reset = HardUpdateAfterSpike(value=ref_len)

    def forward(self, ref_cnt: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ref_cnt[ref_cnt > 0] = ref_cnt[ref_cnt > 0] - 1
            ref_cnt = self.reset(ref_cnt, spike)
        return ref_cnt
