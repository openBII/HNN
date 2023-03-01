# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from typing import Callable


class PrecisionConvert(torch.nn.Module):
    '''精度转换单元

    一般用于负责从ANN转换到SNN的HU中

    Args:
        converter: 精度转换函数
    '''
    def __init__(self, converter: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super(PrecisionConvert, self).__init__()
        self.converter = converter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.converter(x)
        return x