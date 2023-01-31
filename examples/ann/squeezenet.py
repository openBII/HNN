# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import pickle

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class QSqueezeNet(QModel):
    def __init__(
        self,
        version: str = '1_0',
        num_classes: int = 1000
    ) -> None:
        super(QSqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                QConv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                QConv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = QConv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            final_conv,
            nn.ReLU(inplace=True),
            QAdaptiveAvgPool2d((1, 1), 13)
        )

        self.model_name = 'QSqueezeNet'
        self.input_shape = (1, 3, 224, 224)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, record_path=None) -> torch.Tensor:
        if record_path is not None:
            record_dict = {}
            record_dict.update({0: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.features(x)
            x = self.classifier(x)
            record_dict.update(
                {188: x.view(-1).unsqueeze(0).unsqueeze(0).detach().numpy().astype(np.int32)})
            with open(record_path, 'wb') as f:
                pickle.dump(record_dict, f)
            return torch.flatten(x, 1)
        else:
            return self._forward(x)


if __name__ == '__main__':
    model = QSqueezeNet()
    model.execute(is_random_input=True, fix_random_seed=True,
                  result_path='temp/QSqueezeNet/o_0_0_0.dat', export_onnx_path='temp/QSqueezeNet/QSqueezeNet.onnx')
