# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_integrate import QIntegrate
from hnn.grad import FakeQuantize, FakeQuantizeINT28


class QLinear(QIntegrate, torch.nn.Linear):
    '''支持量化的Linear算子

    算子继承自torch.nn.Linear, 基本参数与torch.nn.Linear完全相同, 此处不再赘述

    Args:
        weight_scale: 权重的放缩系数
        is_encoder: 是否作为SNN中的encoder使用
        input_scale: 作为SNN中的encoder使用时对输入的放缩系数
    '''
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, is_encoder=False):
        torch.nn.Linear.__init__(
            self, in_features, out_features, bias, device, dtype)
        QIntegrate.__init__(self, is_encoder=is_encoder)

    def collect_q_params(self):
        '''统计量化参数

        权重的放缩系数直接计算得到
        如果作为encoder使用, 会将算子置于统计量化参数的状态
        '''
        QIntegrate.collect_q_params(self)
        if self.is_encoder:
            self.num_inputs = 0
            self.sum_absmax = 0

    def calculate_q_params(self):
        '''计算量化参数

        计算输入的放缩系数
        '''
        QIntegrate.calculate_q_params(self)
        if self.is_encoder:
            self.input_scale = 128 / (self.sum_absmax / self.num_inputs)
        weight_absmax = self.weight.data.abs().max()
        self.weight_scale = 128 / weight_absmax

    def forward(self, x: torch.Tensor):
        '''前向推理

        Args:
            x: 张量输入

        Returns:
            第一个输出: 张量输出
            第二个输出: 传递给脉冲神经元的量化参数
        '''
        if self.collecting:
            self.num_inputs += 1
            self.sum_absmax += x.data.abs().max()
        if self.is_encoder and self.quantization_mode:
            x = x.mul(self.input_scale).round().clamp(-128, 127)
        out = torch.nn.Linear.forward(self, x)
        if self.quantization_mode:
            out = out.clamp(-134217728, 134217727)  # INT28
        if self.is_encoder:
            return out, self.weight_scale * self.input_scale if (self.weight_scale is not None and self.input_scale is not None) else None
        else:
            return out, self.weight_scale

    def quantize(self):
        QIntegrate.quantize(self)
        self.weight.data = self.weight.data.mul(
            self.weight_scale).round().clamp(-128, 127)  # INT8
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = self.bias.data.mul(
                    self.weight_scale * self.input_scale).round().clamp(-134217728, 134217727)  # INT28
            else:
                self.bias.data = self.bias.data.mul(
                    self.weight_scale).round().clamp(-134217728, 134217727)  # INT28

    def dequantize(self):
        QIntegrate.dequantize(self)
        self.weight.data = self.weight.data.div(self.weight_scale)
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = self.bias.data.div(
                    self.weight_scale * self.input_scale)
            else:
                self.bias.data = self.bias.data.div(self.weight_scale)

    def aware(self):
        if self.quantization_mode:
            self.dequantize()
        QIntegrate.aware(self)
        self.weight.data = FakeQuantize.apply(
            self.weight.data, self.weight_scale)
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = FakeQuantizeINT28.apply(
                    self.bias.data, self.weight_scale * self.input_scale)
            else:
                self.bias.data = FakeQuantizeINT28.apply(
                    self.bias.data, self.weight_scale)