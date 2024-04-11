# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod


class QModule(ABC):
    '''类似于torch.nn.Module, QModel和其他支持量化的算子都继承于QModule类

    Attributes:
        quantization_mode: 表示QModule处于量化模式
        aware_mode: 表示QModule处于量化感知模式
        pretrained: 表示QModule已经加载过预训练模型
    '''
    def __init__(self):
        self.quantization_mode = False
        self.aware_mode = False
        self.pretrained = False

    def quantize(self):
        '''抽象方法, 用于对模型进行量化

        继承QModule的子类中的此方法需要先调用父类的此方法将模型置于正确的状态

        Raises:
            AssertionError: 如果模型已经处于量化状态则调用此方法会报错
        '''
        assert not(self.quantization_mode), 'Model has been quantized'
        self.quantization_mode = True
        self.aware_mode = False

    def aware(self):
        '''抽象方法, 用于对模型进行量化感知训练

        继承QModule的子类中的此方法需要先调用父类的此方法将模型置于正确的状态
        '''
        self.aware_mode = True

    @abstractmethod
    def dequantize(self):
        '''抽象方法, 用于对模型进行反量化, 将量化模型转换成浮点数模型

        继承QModule的子类中的此方法需要先调用父类的此方法将模型置于正确的状态
        '''
        self.quantization_mode = False