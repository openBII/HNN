# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod


class QModule(ABC):
    '''类似于torch.nn.Module, QModel和其他支持量化的算子都继承于QModule类

    Attributes:
        activation_absmax: 静态变量, 用于训练过程中限制激活的范围, default = 1
        quantization_mode: 表示QModule处于量化模式
        aware_mode: 表示QModule处于量化感知模式
        q_params_ready: 表示QModule中的量化参数已经统计完毕
        restricted: 表示QModule处于激活被限制状态
        bit_shift_unit: 硬件上用于实现量化时需要的参数
    '''
    activation_absmax = 1

    def __init__(self):
        self.quantization_mode: bool = False
        self.aware_mode: bool = False
        self.q_params_ready: bool = False
        self.restricted: bool = False
        self.bit_shift_unit: int = None

    @abstractmethod
    def collect_q_params(self):
        '''抽象方法, 用于计算量化参数

        继承QModule的子类中的此方法需要先调用父类的此方法将模型置于正确的状态
        '''
        self.quantization_mode = False
        self.aware_mode = False
        self.q_params_ready = True

    @abstractmethod
    def quantize(self):
        '''抽象方法, 用于对模型进行量化

        继承QModule的子类中的此方法需要先调用父类的此方法将模型置于正确的状态

        Raises:
            AssertionError: 如果模型已经处于量化状态则调用此方法会报错
            AssertionError: 如果模型没有计算得到量化参数则调用此方法会报错
        '''
        assert not(self.quantization_mode), 'Model has been quantized'
        self.quantization_mode = True
        self.aware_mode = False
        assert self.q_params_ready, 'Quantization cannot be executed unless quantization parameters have been collected'

    @abstractmethod
    def aware(self):
        '''抽象方法, 用于对模型进行量化感知训练

        继承QModule的子类中的此方法需要先调用父类的此方法将模型置于正确的状态

        Raises:
            AssertionError: 如果模型没有计算得到量化参数则调用此方法会报错
        '''
        self.aware_mode = True
        assert self.q_params_ready, 'QAT cannot be executed unless quantization parameters have been collected'

    @abstractmethod
    def dequantize(self):
        '''抽象方法, 用于对模型进行反量化, 将量化模型转换成浮点数模型

        继承QModule的子类中的此方法需要先调用父类的此方法将模型置于正确的状态
        '''
        self.quantization_mode = False

    @abstractmethod
    def restrict(self):
        '''抽象方法, 用于对模型的激活值范围进行限制

        继承QModule的子类中的此方法需要先调用父类的此方法将模型置于正确的状态

        Raises:
            AssertionError: 如果模型已经处于量化状态则调用此方法会报错
        '''
        self.restricted = True
        assert not(self.quantization_mode)