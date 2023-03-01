# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_module import QModule
from hnn.grad import FakeQuantize, FakeQuantizeINT28


class QConv2d(QModule, torch.nn.Conv2d):
    '''支持量化的Conv2d算子

    算子继承自torch.nn.Conv2d, 基本参数与torch.nn.Conv2d完全相同, 此处不再赘述

    Args:
        weight_scale: 权重的放缩系数
        is_encoder: 是否作为SNN中的encoder使用
        input_scale: 作为SNN中的encoder使用时对输入的放缩系数
    '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None,
            is_encoder=False):
        torch.nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        QModule.__init__(self)
        self.weight_scale = None
        self.is_encoder = is_encoder
        if is_encoder:
            self.input_scale = None

    def collect_q_params(self):
        '''统计量化参数

        权重的放缩系数直接计算得到
        如果作为encoder使用, 会将算子置于统计量化参数的状态
        '''
        QModule.collect_q_params(self)
        if self.is_encoder:
            self.collecting = True
            self.num_inputs = 0
            self.sum_absmax = 0
        weight_absmax = self.weight.data.abs().max()
        self.weight_scale = 128 / weight_absmax

    def calculate_q_params(self):
        '''计算量化参数

        计算输入的放缩系数
        '''
        self.collecting = False
        self.input_scale = 128 / (self.sum_absmax / self.num_inputs)

    def forward(self, x: torch.Tensor):
        '''前向推理

        Args:
            x: 张量输入

        Returns:
            第一个输出: 张量输出
            第二个输出: 传递给脉冲神经元的量化参数
        '''
        if self.is_encoder and self.q_params_ready:
            if self.collecting:
                self.num_inputs += 1
                self.sum_absmax += x.data.abs().max()
        if self.is_encoder and self.quantization_mode:
            x = x.mul(self.input_scale).round().clamp(-128, 127)
        out = torch.nn.Conv2d.forward(self, x)
        if self.quantization_mode:
            assert not(
                self.aware_mode), 'Quantization mode and QAT mode are mutual exclusive'
            out = out.clamp(-134217728, 134217727)  # INT28
        if self.is_encoder:
            return out, self.weight_scale * self.input_scale if (self.weight_scale is not None and self.input_scale is not None) else None
        else:
            return out, self.weight_scale

    def quantize(self):
        QModule.quantize(self)
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
        QModule.dequantize(self)
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
        QModule.aware(self)
        self.weight.data = FakeQuantize.apply(
            self.weight.data, self.weight_scale)
        if self.bias is not None:
            if self.is_encoder:
                self.bias.data = FakeQuantizeINT28.apply(
                    self.bias.data, self.weight_scale * self.input_scale)
            else:
                self.bias.data = FakeQuantizeINT28.apply(
                    self.bias.data, self.weight_scale)