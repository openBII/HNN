# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.hu import HU
from hnn.hu.a2s_precision_convert import A2SPrecisionConvert
from typing import Callable


class A2SHU(HU):
    '''ANN到SNN转换的HU

    抽象类, 子类必须具有sampler和precision_convert两个部分, 不能包括window_set和window_conv, 一般包括non_linear操作

    Args:
        window_size: 转换后的时间序列长度
        non_linear: 非线性函数
        precision_convert.converter: 精度转换函数
    '''
    def __init__(self, window_size: int, converter: Callable[[torch.Tensor], torch.Tensor],
                 non_linear: torch.nn.Module = None) -> None:
        super(A2SHU, self).__init__(window_size, non_linear)
        self.precision_convert = A2SPrecisionConvert(converter=converter)

    def check(self):
        '''检查A2SHU是否符合基本要求

        子类应该在构造函数最后调用check方法来检查合法性
        '''
        assert (self.window_set is None and self.window_conv is None and
                self.sampler is not None and self.precision_convert is not None)