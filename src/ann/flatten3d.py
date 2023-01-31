# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch


class Flatten3d(torch.nn.Module):
    '''将[N, C, H, W]排布的张量转换成[N, H, W, C]并按照C-order展开到一维
    '''
    def __init__(self) -> None:
        super(Flatten3d, self).__init__()

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        # [N, H, W, C] -> [N, H * W * C]
        x = x.contiguous().view(x.size(0), -1)
        return x