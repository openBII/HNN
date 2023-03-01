# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy

import torch

from hnn.ann.q_conv2d import QConv2d
from hnn.ann.q_linear import QLinear
from hnn.snn.q_conv2d import QConv2d as SQConv2d
from hnn.snn.q_linear import QLinear as SQLinear


def fuse2d_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    assert (conv.training == bn.training), \
        "Conv and BN both must be in the same mode (train or eval)."
    new_conv = torch.nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels,
                               kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
                               dilation=conv.dilation, groups=conv.groups, bias=True, padding_mode=conv.padding_mode)

    if bn.affine:
        gamma = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
        new_conv.weight.data = conv.weight.data * gamma.view(-1, 1, 1, 1)
        if conv.bias is not None:
            new_conv.bias.data = gamma * conv.bias.data - \
                gamma * bn.running_mean + bn.bias.data
        else:
            new_conv.bias.data = bn.bias.data - gamma * bn.running_mean
    else:
        "affine 为 False 的情况, gamma=1, beta=0"
        gamma = 1 / torch.sqrt(bn.running_var + bn.eps)
        new_conv.weight.data = conv.weight.data * gamma
        if conv.bias is not None:
            new_conv.bias.data = gamma * conv.bias.data - gamma * bn.running_mean
        else:
            new_conv.bias.data = - gamma * bn.running_mean
    return new_conv


def fuse1d_linear_bn(linear: torch.nn.Linear, bn: torch.nn.BatchNorm1d):
    assert (linear.training == bn.training), \
        "Linear and BN both must be in the same mode (train or eval)."
    new_linear = torch.nn.Linear(
        in_features=linear.in_features, out_features=linear.out_features, bias=True)

    if bn.affine:
        gamma = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
        new_linear.weight.data = linear.weight.data * gamma.view(-1, 1)
        if linear.bias is not None:
            new_linear.bias.data = gamma * linear.bias.data - \
                gamma * bn.running_mean + bn.bias.data
        else:
            new_linear.bias.data = bn.bias.data - gamma * bn.running_mean
    else:
        "affine 为 False 的情况, gamma=1, beta=0"
        gamma = 1 / torch.sqrt(bn.running_var + bn.eps)
        new_linear.weight.data = linear.weight.data * gamma
        if linear.bias is not None:
            new_linear.bias.data = gamma * linear.bias.data - gamma * bn.running_mean
        else:
            new_linear.bias.data = - gamma * bn.running_mean
    return new_linear


# Generalization of getattr
def _get_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


# Generalization of setattr
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    setattr(cur_mod, tokens[-1], module)


def fuse_known_modules(mod_list):
    OP_LIST_TO_FUSER_METHOD = {
        # (torch.nn.Conv1d, torch.nn.BatchNorm1d): fuse_conv_bn,
        # (torch.nn.Conv1d, torch.nn.BatchNorm1d, torch.nn.ReLU): fuse_conv_bn_relu,
        (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse2d_conv_bn,
        (torch.nn.Linear, torch.nn.BatchNorm1d): fuse1d_linear_bn,
        (QConv2d, torch.nn.BatchNorm2d): fuse2d_conv_bn,
        (QLinear, torch.nn.BatchNorm1d): fuse1d_linear_bn,
        (SQConv2d, torch.nn.BatchNorm2d): fuse2d_conv_bn,
        (SQLinear, torch.nn.BatchNorm1d): fuse1d_linear_bn,
        # (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU): fuse_conv_bn_relu,
        # (torch.nn.Conv3d, torch.nn.BatchNorm3d): fuse_conv_bn,
        # (torch.nn.Conv3d, torch.nn.BatchNorm3d, torch.nn.ReLU): fuse_conv_bn_relu,
        # (torch.nn.Conv1d, torch.nn.ReLU): torch.nn.intrinsic.ConvReLU1d,
        # (torch.nn.Conv2d, torch.nn.ReLU): torch.nn.intrinsic.ConvReLU2d,
        # (torch.nn.Conv3d, torch.nn.ReLU): torch.nn.intrinsic.ConvReLU3d,
        # (torch.nn.Linear, torch.nn.ReLU): torch.nn.intrinsic.LinearReLU,
        # (torch.nn.BatchNorm2d, torch.nn.ReLU): torch.nn.intrinsic.BNReLU2d,
        # (torch.nn.BatchNorm3d, torch.nn.ReLU): torch.nn.intrinsic.BNReLU3d,
    }

    types = tuple(type(m) for m in mod_list)
    fuser_method = OP_LIST_TO_FUSER_METHOD.get(types, None)
    if fuser_method is None:
        raise NotImplementedError("Cannot fuse modules: {}".format(types))
    new_mod = [None] * len(mod_list)
    new_mod[0] = fuser_method(*mod_list)

    for i in range(1, len(mod_list)):
        new_mod[i] = torch.nn.Identity()
        new_mod[i].training = mod_list[0].training

    return new_mod


def _fuse_modules(model, modules_to_fuse, fuser_func=fuse_known_modules):
    mod_list = []
    for item in modules_to_fuse:
        mod_list.append(_get_module(model, item))

    # Fuse list of modules
    new_mod_list = fuser_func(mod_list)

    # Replace original module list with fused module list
    for i, item in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])


def fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=fuse_known_modules):
    if not inplace:
        model = copy.deepcopy(model)

    if all(isinstance(module_element, str) for module_element in modules_to_fuse):
        # Handle case of modules_to_fuse being a list
        _fuse_modules(model, modules_to_fuse, fuser_func)
    else:
        # Handle case of modules_to_fuse being a list of lists
        for module_list in modules_to_fuse:
            _fuse_modules(model, module_list, fuser_func)
    return model
