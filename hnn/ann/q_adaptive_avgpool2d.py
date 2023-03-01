# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import math
from hnn.ann.q_module import QModule
from hnn.grad import FakeQuantizeFloor


class QAdaptiveAvgPool2d(QModule, torch.nn.AdaptiveAvgPool2d):
    '''支持量化的平均池化算子

    算子继承自torch.nn.AdaptiveAvgPool2d, 基本参数与torch.nn.AdaptiveAvgPool2d完全相同, 此处不再赘述
    目前只考虑了整个模型中只出现一个平均池化

    Args:
        bit_shift: 完成定点数计算后需要的量化参数
        absmax: 对输出激活进行限制时的范围
        kernel_size: 池化窗的大小
        is_last_node: 是否是最后一个算子
    '''

    def __init__(self, output_size, kernel_size, is_last_node=False):
        torch.nn.AdaptiveAvgPool2d.__init__(self, output_size)
        QModule.__init__(self)
        self.bit_shift = None
        self.absmax = None
        self.kernel_size = kernel_size
        self.is_last_node = is_last_node

    def collect_q_params(self, bit_shift_unit):
        QModule.collect_q_params(self)
        self.bit_shift = bit_shift_unit * round(math.log(self.kernel_size, 2))
        self.absmax = 2 ** self.bit_shift / self.kernel_size ** 2

    def forward(self, x: torch.Tensor):
        if self.restricted:
            x = x.clamp(-QModule.activation_absmax, QModule.activation_absmax)
            QModule.activation_absmax = self.absmax
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = FakeQuantizeFloor.apply(x, 128 / QModule.activation_absmax)
            QModule.activation_absmax = self.absmax
        out = torch.nn.AdaptiveAvgPool2d.forward(self, x)
        if self.quantization_mode and not(self.is_last_node):
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.mul(self.kernel_size ** 2).clamp(-2147483648, 2147483647).div(2 **
                                                                                   self.bit_shift).floor().clamp(-128, 127)
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
        '''平均池化量化参数的计算在restrict方法中完成

        y = (x1 + x2 + ... + x_kernel_size) / kernel_size^2
        128y = (128x1 + 128x2 + ... + 128x_kernel_size) / kernel_size^2
        n = round((log_2 kernel_size^2) / 2)
        bit_shift = n * bit_shift_unit
        128 * kernel_size^2 / 2^bit_shift y = (128x1 + 128x2 + ... + 128x_kernel_size) / 2^bit_shift
        '''
        QModule.restrict(self)
        self.bit_shift_unit = bit_shift_unit
        self.bit_shift = bit_shift_unit * round(math.log(self.kernel_size, 2))
        self.absmax = 2 ** self.bit_shift / self.kernel_size ** 2