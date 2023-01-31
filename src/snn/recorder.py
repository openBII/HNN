# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch


class Recorder(torch.autograd.Function):
    '''记录脉冲神经元的各种参数信息和标识脉冲神经元

    抽象类, 脉冲神经元需要根据需求继承Recorder类, 主要用于记录脉冲神经元的各种参数信息以及在计算图中起到标识脉冲神经元的作用
    '''
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value):
        # FIXME(huanyu): 这里有个pytorch的bug没有修复, 正常应该通过setType()设置形状, 但shape inference还是会missing
        # 这issue好几个月前就提了pytorch还没有修复烦死了😡
        return g.op("snn::Record", input)