# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.hu import HU
from hnn.hu.window_set import WindowSet


class S2AHU(HU):
    '''SNN到ANN转换的HU

    抽象类, 子类必须具有window_set和window_conv两个部分, 不能包括sampler和precision_convert, 一般包括non_linear操作

    Args:
        window_size: 时间窗大小
        non_linear: 非线性函数
        window_set: 设置时间窗
    '''
    def __init__(self, window_size: int, non_linear: torch.nn.Module = None) -> None:
        super(S2AHU, self).__init__(window_size, non_linear)
        self.window_set = WindowSet(size=window_size)

    def check(self):
        '''检查S2AHU是否符合基本要求

        子类应该在构造函数最后调用check方法来检查合法性
        '''
        assert (self.window_set is not None and self.window_conv is not None and
                self.sampler is None and self.precision_convert is None)