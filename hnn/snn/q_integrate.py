# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import logging
from abc import ABC, abstractmethod
from hnn.snn.q_module import QModule


class QIntegrate(QModule, ABC):
    '''类似于torch.nn.Module, QModel和其他支持量化的算子都继承于QModule类

    Attributes:
        quantization_mode: 表示QModule处于量化模式
        aware_mode: 表示QModule处于量化感知模式
        q_params_ready: 表示QModule中的量化参数已经统计完毕
        pretrained: 表示QModule已经加载过预训练模型
    '''
    def __init__(self, is_encoder: bool):
        super().__init__()
        self.q_params_ready = False
        self.is_encoder = is_encoder
        self.collecting = False
        self.weight_scale = None
        self.bias_scale = None
        if self.is_encoder:
            self.input_scale = None

    @abstractmethod
    def collect_q_params(self):
        '''统计量化参数

        继承QIntegrate的子类中的此方法需要先调用父类的此方法将模型置于正确的状态
        如果算子作为encoder使用, 会将算子置于统计量化参数的状态
        '''
        self.quantization_mode = False
        self.aware_mode = False
        if self.is_encoder:
            self.collecting = True
        if not(self.pretrained):
            logging.warning(
                'Collecting quantization parameters usually requires a pretrained model')

    @abstractmethod
    def calculate_q_params(self):
        '''抽象方法, 用于计算量化参数

        继承QIntegrate的子类中的此方法需要先调用父类的此方法将模型置于正确的状态
        '''
        self.quantization_mode = False
        self.aware_mode = False
        self.q_params_ready = True
        self.collecting = False

    @abstractmethod
    def quantize(self):
        '''抽象方法, 用于对模型进行量化

        继承QIntegrate的子类中的此方法需要先调用父类的此方法将模型置于正确的状态

        Raises:
            AssertionError: 如果模型已经处于量化状态则调用此方法会报错
        '''
        QModule.quantize(self)
        assert self.q_params_ready, 'Quantization cannot be executed unless quantization parameters have been collected'

    @abstractmethod
    def aware(self):
        '''抽象方法, 用于对模型进行量化感知训练

        继承QModule的子类中的此方法需要先调用父类的此方法将模型置于正确的状态

        Raises:
            AssertionError: 如果模型没有计算得到量化参数则调用此方法会报错
        '''
        QModule.aware(self)
        assert self.q_params_ready, 'QAT cannot be executed unless quantization parameters have been collected'
