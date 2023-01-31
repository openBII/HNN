# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import pickle
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from src.ann.q_adaptive_avgpool2d import QAdaptiveAvgPool2d
from src.ann.q_conv2d import QConv2d
from src.ann.q_model import QModel


class Fire(nn.Module):
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = QConv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = QConv2d(squeeze_planes, expand1x1_planes,
                                 kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = QConv2d(squeeze_planes, expand3x3_planes,
                                 kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

    def forward(self, x: torch.Tensor, record_dict: Dict = None, block_ids=None) -> torch.Tensor:
        if record_dict is not None:
            x = self.squeeze_activation(self.squeeze(x))
            record_dict.update({block_ids[0]: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x0 = self.expand1x1_activation(self.expand1x1(x))
            record_dict.update({block_ids[1]: x0.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x1 = self.expand3x3_activation(self.expand3x3(x))
            record_dict.update({block_ids[2]: x1.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = torch.cat([x0, x1], 1)
            record_dict.update({block_ids[3]: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            return x
        else:
            return self._forward(x)


class QSmallSqueezeNet(QModel):
    def __init__(
        self,
        num_classes: int = 10
    ) -> None:
        super(QSmallSqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.conv0 = QConv2d(3, 64, kernel_size=3, stride=2)
        self.maxpool0 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.fire0 = Fire(64, 16, 64, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire1 = Fire(128, 32, 256, 256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Final convolution is initialized differently from the rest
        final_conv = QConv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            final_conv,
            nn.ReLU(inplace=True),
            QAdaptiveAvgPool2d((1, 1), 13)
        )

        self.model_name = 'QSmallSqueezeNet'
        self.input_shape = (1, 3, 224, 224)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.maxpool0(x)
        x = self.fire0(x)
        x = self.maxpool1(x)
        x = self.fire1(x)
        x = self.maxpool2(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, record_path=None) -> torch.Tensor:
        if record_path is not None:
            record_dict = {}
            record_dict.update({0: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.conv0(x)
            x = self.maxpool0(x)
            record_dict.update({8: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.fire0(x, record_dict, [13, 19, 25, 29])
            x = self.maxpool1(x)
            record_dict.update({31: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.fire1(x)
            x = self.maxpool2(x)
            record_dict.update({54: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.classifier(x)
            record_dict.update(
                {62: x.view(-1).unsqueeze(0).unsqueeze(0).detach().numpy().astype(np.int32)})
            with open(record_path, 'wb') as f:
                pickle.dump(record_dict, f)
            return torch.flatten(x, 1)
        else:
            return self._forward(x)


if __name__ == '__main__':
    model = QSmallSqueezeNet()
    model.execute(is_random_input=True, fix_random_seed=True,
                  result_path='temp/QSmallSqueezeNet/o_0_0_0.dat', export_onnx_path='temp/QSmallSqueezeNet/QSmallSqueezeNet.onnx')
