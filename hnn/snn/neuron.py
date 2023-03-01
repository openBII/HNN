# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.q_model import QModel


class Neuron(QModel):
    '''神经元基类

    需要导出成ONNX模型的SNN中的神经元需要继承Neuron类
    主要功能为在进行实际的神经元计算前插入Recorder结点, 通过Recorder结点标识神经元并且记录神经元参数, 然后完成正常的神经元计算

    Args:
        recorder: 继承自Recorder类, 不完成实际计算, 主要用于标识神经元和记录各种参数
        T: SNN的时间步
    '''
    def __init__(self, recorder, T):
        super(Neuron, self).__init__()
        self.recorder = recorder
        self.T = T

    def record(self, x: torch.Tensor) -> torch.Tensor:
        return self.recorder.apply(x)

    def _forward(self, *args):
        return args

    def forward(self, *args):
        return self._forward(self.record(args[0]), *args[1:])