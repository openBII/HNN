# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from typing import Callable
from hnn.hu.precision_convert import PrecisionConvert


class A2SPrecisionConvert(PrecisionConvert):
    '''用于ANN到SNN的精度转换单元

    Args:
        converter: 精度转换函数
    '''
    def __init__(self, converter: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super(A2SPrecisionConvert, self).__init__(converter=converter)