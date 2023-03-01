# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import logging
from typing import Dict
from hnn.snn.q_module import QModule


class QModel(QModule, torch.nn.Module):
    '''支持量化的网络模型需要继承于QModel类

    Attributes:
        T: SNN需要执行的时间步
    '''
    def __init__(self, time_window_size: int = None):
        QModule.__init__(self)
        torch.nn.Module.__init__(self)
        self.T = time_window_size

    def collect_q_params(self):
        '''递归地调用网络中所有算子的collect_q_params方法

        只有SNN中负责完成Integrate操作的算子, 例如卷积和全连接, 需要重载collect_q_params方法
        '''
        if not(self.pretrained):
            logging.warning(
                'Collecting quantization parameters usually requires a pretrained model')
        QModule.collect_q_params(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                if hasattr(module, 'collect_q_params'):
                    module.collect_q_params()

    def calculate_q_params(self):
        '''计算量化参数

        只有SNN中负责完成Integrate操作的算子, 例如卷积和全连接, 需要重载calculate_q_params方法
        '''
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                if hasattr(module, 'is_encoder'):
                    if module.is_encoder:
                        module.calculate_q_params()

    def load_model(self, model_path: str, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        '''加载浮点数预训练模型

        Args:
            model_path: 预训练模型保存路径
            devive: 模型的map_location
        '''
        state_dict = torch.load(model_path, map_location=map_location)
        self.load_state_dict(state_dict)
        self.pretrained = True

    def save_quantized_model(self, checkpoint_path, others: Dict = None):
        '''保存量化模型

        保存的checkpoint为一个字典, 默认包含模型本身的state_dict, 量化参数和脉冲神经元的各种参数, key分别为model和q_params

        Args:
            checkpoint_path: checkpoint保存路径
            others: 如果有除了state_dict和量化参数以外需要保存的内容可以通过others传入并保存
        '''
        q_params_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                q_params_dict[name] = {}
                q_params_dict[name]['weight_scale'] = module.weight_scale
                if hasattr(module, 'if_node'):
                    q_params_dict[name]['v_th_0'] = module.if_node.fire.v_th
                    q_params_dict[name]['v_reset'] = module.if_node.reset.value
                    q_params_dict[name]['v_init'] = module.if_node.accumulate.v_init
                if hasattr(module, 'v_leaky'):
                    q_params_dict[name]['v_leaky_beta'] = module.v_leaky.beta
                    if module.v_leaky.adpt_en:
                        q_params_dict[name]['v_leaky_alpha'] = module.v_leaky.alpha
                else:
                    if module.is_encoder:
                        q_params_dict[name]['input_scale'] = module.input_scale
        checkpoint = {
            'model': self.state_dict(),
            'q_params': q_params_dict
        }
        if others is not None:
            checkpoint.update(others)
        torch.save(checkpoint, checkpoint_path)

    def load_quantized_model(self, checkpoint_path: str, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        '''加载量化模型

        此方法会自动加载state_dict, 量化参数和脉冲神经元的各种参数, 其他保存到checkpoint中需要加载的内容需要用户手动加载

        Args:
            checkpoint_path: checkpoint保存路径
            device: 模型的map_location

        Returns:
            checkpoint: 通过torch.load加载的字典
        '''
        self.q_params_ready = True
        self.quantization_mode = True
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(checkpoint['model'])
        q_params_dict = checkpoint['q_params']
        for name, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.q_params_ready = True
                module.quantization_mode = True
                module.pretrained = True
                module.weight_scale = q_params_dict[name]['weight_scale']
                if hasattr(module, 'if_node'):
                    module.if_node.fire.v_th = q_params_dict[name]['v_th_0']
                    module.if_node.reset.value = q_params_dict[name]['v_reset']
                    module.if_node.accumulate.v_init = q_params_dict[name]['v_reset']
                if hasattr(module, 'v_leaky'):
                    module.v_leaky.beta = q_params_dict[name]['v_leaky_beta']
                    if 'v_leaky_alpha' in q_params_dict[name]:
                        module.v_leaky.alpha = q_params_dict[name]['v_leaky_alpha']
                else:
                    if module.is_encoder:
                        module.input_scale = q_params_dict[name]['input_scale']
        self.pretrained = True
        return checkpoint

    def quantize(self):
        '''对网络进行量化

        此方法只对SNN中负责Integrate的算子进行量化
        '''
        QModule.quantize(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.quantize()

    def aware(self, *dummy_inputs):
        '''将网络置于量化感知训练的模式

        由于SNN的脉冲神经元的量化参数需要负责Integrate的算子传递, 所以先执行一次前向推理过程
        如果网络处于量化模式, 则会先执行反量化操作
        '''
        self.forward(*dummy_inputs)
        if self.quantization_mode:
            self.dequantize()
        QModule.aware(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.aware()

    def dequantize(self):
        '''对网络进行反量化
        '''
        QModule.dequantize(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.dequantize()

    def refresh(self):
        '''连续两次执行推理过程之间需要调用refresh方法

        refresh方法保证第二次推理过程中可以自动对输入进行量化, 但不会重复对脉冲神经元参数重复进行量化
        '''
        for _, module in self.named_modules():
            if isinstance(module, QModule):
                if hasattr(module, 'first_time'):
                    module.first_time = True
                    module.freeze = True