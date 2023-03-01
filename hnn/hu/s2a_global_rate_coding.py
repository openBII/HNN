# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.s2ahu import S2AHU
from hnn.hu.global_average_window_conv import GlobalAverageWindowConv


class S2AGlobalRateCoding(S2AHU):
    '''使用全局平均时间窗卷积的S2AHU
    '''
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        super(S2AGlobalRateCoding, self).__init__(window_size, non_linear)
        self.window_conv = GlobalAverageWindowConv(window_size=window_size)
        self.check()