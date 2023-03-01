# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import os

import onnx
import onnxsim
import torch
from onnx.shape_inference import infer_shapes

from hnn.network_type import NetworkType


def onnx_export(
    model: torch.nn.Module, input, output_path,
    model_path=None,
    reserve_control_flow=False,
    network_type: NetworkType = NetworkType.ANN
):
    if network_type == NetworkType.ANN:
        if model_path is not None:
            if hasattr(model, 'load_quantized_model'):
                model.load_quantized_model(
                    checkpoint_path=model_path,
                    device=torch.device('cpu')
                )
            else:
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
        if reserve_control_flow:
            model = torch.jit.script(model)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.onnx.export(model=model, args=input, f=output_path,
                          keep_initializers_as_inputs=True,
                          do_constant_folding=True)
        onnx_model = onnx.load(output_path)
        onnx_model, _ = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, output_path)
    elif network_type == NetworkType.SNN:
        if model_path is not None:
            if hasattr(model, 'load_quantized_model'):
                model.load_quantized_model(
                    checkpoint_path=model_path,
                    device=torch.device('cpu')
                )
            else:
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
        if reserve_control_flow:
            model = torch.jit.script(model)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.onnx.export(model=model, args=input, f=output_path,
                          keep_initializers_as_inputs=True,
                          do_constant_folding=False,
                          custom_opsets={'snn': 1})
        # SNN中由于存在自定义算子无法调用onnx-simplifier, 需要在onnxruntime中实现自定义算子
        onnx_model = onnx.load(output_path)
        onnx.save(infer_shapes(onnx_model), output_path)
    else:
        raise NotImplementedError(network_type.name + 'has not been supported')

    return onnx.load(output_path)
