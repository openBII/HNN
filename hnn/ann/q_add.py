# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.ann.q_module import QModule
from hnn.grad import FakeQuantizeFloor


class QAdd(QModule, torch.nn.Module):
    '''支持量化的张量加算子

    Args:
        bit_shift: 完成定点数计算后需要的量化参数
        is_last_node: 是否是最后一个算子
    '''
    def __init__(self, is_last_node=False):
        torch.nn.Module.__init__(self)
        QModule.__init__(self)
        self.bit_shift = None
        self.is_last_node = is_last_node

    def collect_q_params(self, _):
        '''计算张量加的量化参数

        如果采用先限制激活再量化的方法, 则量化参数为固定值, 在完成定点数张量加之后不需要进行特殊处理
        '''
        QModule.collect_q_params(self)
        self.bit_shift = 0

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.restricted:
            x = x.clamp(-QModule.activation_absmax, QModule.activation_absmax)
            y = y.clamp(-QModule.activation_absmax, QModule.activation_absmax)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = FakeQuantizeFloor.apply(x, 128 / QModule.activation_absmax)
            y = FakeQuantizeFloor.apply(y, 128 / QModule.activation_absmax)
        out = x + y
        if self.quantization_mode and not(self.is_last_node):
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.clamp(-2147483648,
                            2147483647).div(2 ** self.bit_shift).floor().clamp(-128, 127)
        if self.is_last_node:
            out = out.clamp(-2147483648, 2147483647)
        return out

    def quantize(self):
        QModule.quantize(self)

    def dequantize(self):
        QModule.dequantize(self)

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)

    def restrict(self, bit_shift_unit):
        QModule.restrict(self)
        self.bit_shift_unit = bit_shift_unit