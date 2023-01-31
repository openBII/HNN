# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import pickle
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.ann.q_adaptive_avgpool2d import QAdaptiveAvgPool2d
from src.ann.q_add import QAdd
from src.ann.q_conv2d import QConv2d
from src.ann.q_linear import QLinear
from src.ann.q_model import QModel


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias=False) \
        -> QConv2d:
    """3x3 convolution with padding"""
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias=False) -> QConv2d:
    """1x1 convolution"""
    return QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            batch_norm: bool = True,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.batch_norm = batch_norm
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(
            inplanes, width, stride=stride, bias=not batch_norm)
        self.relu1 = nn.ReLU(inplace=True)

        if self.batch_norm:
            self.bn1 = norm_layer(width)
            self.bn2 = norm_layer(width)
            self.bn3 = norm_layer(planes * self.expansion)
        self.conv2 = conv3x3(width, width, 1, groups,
                             dilation, bias=not batch_norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(
            width, planes * self.expansion, bias=not batch_norm)

        self.downsample = downsample

        self.add = QAdd()

        self.relu3 = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x: Tensor, record_dict: Dict = None, block_ids=None) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu1(out)
        if record_dict is not None:
            record_dict.update({block_ids[0]: out.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.relu2(out)
        if record_dict is not None:
            record_dict.update({block_ids[1]: out.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})

        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)
        if record_dict is not None:
            record_dict.update({block_ids[2]: out.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})

        if self.downsample is not None:
            identity = self.downsample(x)
            if record_dict is not None:
                record_dict.update({block_ids[3]: identity.squeeze(
                    0).permute(1, 2, 0).detach().numpy().astype(np.int32)})

        out = self.add(out, identity)
        out = self.relu3(out)

        return out


class QResNet(QModel):

    def __init__(
            self,
            block: Type[Bottleneck],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            batch_norm: bool = True,
    ) -> None:
        super(QResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.batch_norm = batch_norm
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = QConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=(3, 3),
                             bias=not batch_norm)
        if self.batch_norm:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = QAdaptiveAvgPool2d((1, 1), 7)
        self.fc = QLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, QConv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)

        self.model_name = 'QResNext'
        self.input_shape = (1, 3, 224, 224)

    def _make_layer(self, block: Type[Bottleneck], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.batch_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion,
                            stride, not self.batch_norm),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion,
                            stride, not self.batch_norm)
                )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer, batch_norm=self.batch_norm)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, batch_norm=self.batch_norm))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1[0](x)
        x = self.layer1[1](x)
        x = self.layer1[2](x)
        x = self.layer2[0](x)
        x = self.layer2[1](x)
        x = self.layer2[2](x)
        x = self.layer2[3](x)
        x = self.layer3[0](x)
        x = self.layer3[1](x)
        x = self.layer3[2](x)
        x = self.layer3[3](x)
        x = self.layer3[4](x)
        x = self.layer3[5](x)
        x = self.layer4[0](x)
        x = self.layer4[1](x)
        x = self.layer4[2](x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, record_path=None) -> Tensor:
        if record_path is not None:
            record_dict = {}
            record_dict.update({0: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.conv1(x)
            if self.batch_norm:
                x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            record_dict.update({7: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})

            # [13, 19, 374, 376]
            x = self.layer1[0](x, record_dict, [13, 19, 28, 29])
            record_dict.update({32: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.layer1[1](x)
            x = self.layer1[2](x)
            record_dict.update({74: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.layer2[0](x)
            x = self.layer2[1](x)
            x = self.layer2[2](x)
            x = self.layer2[3](x)
            record_dict.update({162: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.layer3[0](x)
            x = self.layer3[1](x)
            x = self.layer3[2](x)
            x = self.layer3[3](x)
            x = self.layer3[4](x)
            x = self.layer3[5](x)
            record_dict.update({292: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.layer4[0](x)
            x = self.layer4[1](x)
            x = self.layer4[2](x)
            record_dict.update({359: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})

            x = self.avgpool(x)
            record_dict.update(
                {506: x.view(-1).detach().numpy().astype(np.int32)})

            x = torch.flatten(x, 1)
            x = self.fc(x)
            record_dict.update(
                {366: x.view(-1).detach().numpy().astype(np.int32)})
            with open(record_path, 'wb') as f:
                pickle.dump(record_dict, f)
            return x
        else:
            return self._forward_impl(x)


def _qresnet(
        block: Type[Bottleneck],
        layers: List[int],
        batch_norm: bool = True,
        **kwargs: Any
) -> QResNet:
    model = QResNet(block, layers, batch_norm=batch_norm, **kwargs)
    return model


def qresnext50_32x4d(batch_norm: bool = False, **kwargs: Any) -> QResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _qresnet(Bottleneck, [3, 4, 6, 3], batch_norm=batch_norm, **kwargs)


if __name__ == '__main__':
    model = qresnext50_32x4d()
    model.execute(is_random_input=True, fix_random_seed=True,
                  result_path='temp/QResNext50_32x4d/o_0_0_0.dat', export_onnx_path='temp/QResNext50_32x4d/QResNet50_32x4d.onnx')
