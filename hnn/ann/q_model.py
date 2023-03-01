# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import os
import torch
import logging
from hnn.ann.q_module import QModule
from hnn.utils import setup_random_seed, get_int8_tensor
from typing import Dict
from hnn.onnx_export_pass import onnx_export


class QModel(QModule, torch.nn.Module):
    '''支持量化的网络模型需要继承于QModel类

    Attributes:
        pretrained: 代表模型已加载预训练模型
    '''
    def __init__(self, bit_shift_unit: int = 2, activation_absmax: int = 1):
        QModule.__init__(self)
        torch.nn.Module.__init__(self)
        self.bit_shift_unit = bit_shift_unit
        self.pretrained: bool = False
        QModule.activation_absmax = activation_absmax

    def collect_q_params(self):
        '''自动计算网络中所有算子的量化参数
        '''
        if not(self.pretrained):
            logging.warning(
                'Collecting quantization parameters usually requires a pretrained model')
        QModule.collect_q_params(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.collect_q_params(self.bit_shift_unit)

    def load_model(self, model_path: str, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        '''加载浮点数预训练模型

        Args:
            model_path: 预训练模型保存路径
            devive: 模型的map_location
        '''
        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)
        self.pretrained = True

    def save_quantized_model(self, checkpoint_path: str, others: Dict = None):
        '''保存量化模型, 包括模型的state_dict和量化参数

        保存的checkpoint为一个字典, 默认包含模型本身的state_dict和量化参数, key分别为model和q_params

        Args:
            checkpoint_path: checkpoint保存路径
            others: 如果有除了state_dict和量化参数以外需要保存的内容可以通过others传入并保存
        '''
        q_params_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                q_params_dict[name] = module.bit_shift
        checkpoint = {
            'model': self.state_dict(),
            'q_params': q_params_dict
        }
        if others is not None:
            checkpoint.update(others)
        torch.save(checkpoint, checkpoint_path)

    def load_quantized_model(self, checkpoint_path: str, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict:
        '''加载预训练量化模型

        此方法会自动加载state_dict和量化参数, 其他保存到checkpoint中需要加载的内容需要用户手动加载

        Args:
            checkpoint_path: checkpoint保存路径
            device: 模型的map_location

        Returns:
            checkpoint: 通过torch.load加载的字典
        '''
        self.q_params_ready = True
        self.quantization_mode = True
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model']
        self.load_state_dict(state_dict)
        q_params_dict = checkpoint['q_params']
        last_module = None
        for name, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.bit_shift = q_params_dict[name]
                module.q_params_ready = True
                module.quantization_mode = True
                module.weight_scale = 2 ** module.bit_shift
                last_module = module
        last_module.is_last_node = True
        self.pretrained = True
        return checkpoint

    def quantize(self, keep_output_precision=True):
        '''对网络进行量化
        '''
        QModule.quantize(self)
        last_module = None
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.quantize()
                last_module = module
        if keep_output_precision:
            last_module.is_last_node = True

    def aware(self):
        '''将网络置于量化感知训练的模式

        如果网络处于量化模式, 则会先执行反量化操作
        '''
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

    def restrict(self):
        '''将网络置于激活值受限的状态
        '''
        QModule.restrict(self)
        for _, module in self.named_modules():
            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                module.restrict(self.bit_shift_unit)

    def execute(self, is_random_input: bool = True, fix_random_seed: bool = True, input_data_path: str = None,
                pre_model_path: str = None, result_path: str = None, export_onnx_path: str = None):
        '''执行模型的完整流程

        Args:
            is_random_input: 是否随机输入
            fix_random_seed: is_random_input为是的话是否固定随机数种子
            input_data_path: is_random_input为否的话需要给出输入文件的路径
            pre_model_path: 预训练模型文件路径
            result_path: 模型推理的输出保存路径
            export_onnx_path: ONNX文件路径
        '''
        # 模型执行的基本准备
        if fix_random_seed:
            random_seed = sum(ord(c) for c in self.model_name)
            setup_random_seed(random_seed)

        # 创建输入数据
        x = None
        if input_data_path is not None:
            pass
        if is_random_input:
            x = get_int8_tensor(self.input_shape)

        # 量化相关设置
        self.collect_q_params()  # 设置量化参数
        self.quantize()   # 将模型置于量化模式

        # 加载预训练的量化模型
        if pre_model_path is not None:
            self.load_quantized_model(
                checkpoint_path=pre_model_path,
                device=torch.device('cpu')
            )

        # 推理模型产生结果
        y = self.forward(x)
        if result_path is not None:
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            y.detach().numpy().tofile(result_path)

        # 导出ONNX模型
        if export_onnx_path is not None:
            os.makedirs(os.path.dirname(export_onnx_path), exist_ok=True)
            onnx_export(model=self, input=x, output_path=export_onnx_path)