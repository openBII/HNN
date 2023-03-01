# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import math
from hnn.ann.q_module import QModule
from hnn.grad import FakeQuantizeFloor, FakeQuantize, FakeQuantizeINT32


class QLinear(QModule, torch.nn.Linear):
    '''支持量化的Linear算子

    算子继承自torch.nn.Linear, 基本参数与torch.nn.Linear完全相同, 此处不再赘述

    Args:
        weight_scale: 权重的放缩系数
        bit_shift: 完成定点数计算后需要的量化参数
        is_last_node: 是否是最后一个算子
    '''
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, is_last_node=False):
        torch.nn.Linear.__init__(
            self, in_features, out_features, bias, device, dtype)
        QModule.__init__(self)
        self.weight_scale = None
        self.bit_shift = None
        self.is_last_node = is_last_node

    def collect_q_params(self, bit_shift_unit):
        '''全连接中计算量化参数

        weight_absmax * weight_scale = 128
        weight_scale = 2^(bit_shift_unit * n)
        bit_shift = bit_shift_unit * n = log_2 (128 / weight_absmax)
        n = round(log_2 (128 / weight_absmax) / bit_shift_unit)
        这里取整方法可以有很多
        '''
        QModule.collect_q_params(self)
        weight_absmax = self.weight.data.abs().max()
        temp = math.log(128 / weight_absmax, 2) / bit_shift_unit
        if temp - math.floor(temp) >= 0.75:  # 经验公式
            n = math.ceil(temp)
        else:
            n = math.floor(temp)
        self.bit_shift = bit_shift_unit * n
        self.weight_scale = 2 ** self.bit_shift

    def forward(self, x: torch.Tensor):
        if self.restricted:
            x = x.clamp(-QModule.activation_absmax, QModule.activation_absmax)
        if self.aware_mode:
            assert not(
                self.quantization_mode), 'Quantization mode and QAT mode are mutual exclusive'
            x = FakeQuantizeFloor.apply(x, 128 / QModule.activation_absmax)
        out = torch.nn.Linear.forward(self, x)
        if self.quantization_mode and not(self.is_last_node):
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.clamp(-2147483648,
                            2147483647).div(self.weight_scale).floor().clamp(-128, 127)
        if self.is_last_node:
            out = out.clamp(-2147483648, 2147483647)
        return out

    def quantize(self):
        QModule.quantize(self)
        self.weight.data = self.weight.data.mul(
            self.weight_scale).round().clamp(-128, 127)  # INT8
        self.bias.data = self.bias.data.mul(
            self.weight_scale * 128 / QModule.activation_absmax).round().clamp(-2147483648, 2147483647)  # INT32

    def dequantize(self):
        QModule.dequantize(self)
        self.weight.data = self.weight.data.div(self.weight_scale)
        self.bias.data = self.bias.data.div(
            self.weight_scale * 128 / QModule.activation_absmax)

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        self.weight.data = FakeQuantize.apply(
            self.weight.data, self.weight_scale)
        self.bias.data = FakeQuantizeINT32.apply(
            self.bias.data, self.weight_scale * 128 / QModule.activation_absmax)

    def restrict(self, bit_shift_unit):
        QModule.restrict(self)
        self.bit_shift_unit = bit_shift_unit