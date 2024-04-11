# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from hnn.snn.q_module import QModule


class QDynamics(QModule):
    '''类似于torch.nn.Module, QModel和其他支持量化的算子都继承于QModule类

    Attributes:
        scale: 量化参数, 用于对脉冲神经元参数进行放缩
        first_time: 只有初次执行时才会对输入膜电位进行量化, 后续时间步执行时的输入膜电位已经被量化过不需要再被量化
        freeze: 脉冲神经元参数是否处于冻结状态
    '''
    def __init__(self):
        self.scale = None
        self.first_time = True
        self.freeze = False