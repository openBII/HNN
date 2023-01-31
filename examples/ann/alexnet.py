# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import torch.nn as nn
from src.ann.q_conv2d import QConv2d
from src.ann.q_linear import QLinear
from src.ann.q_model import QModel


class QAlexNet(QModel):
    def __init__(self, num_classes=1000):
        super(QAlexNet, self).__init__()
        self.conv0 = QConv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool0 = nn.MaxPool2d(3, stride=2)

        self.conv1 = QConv2d(64, 192, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = QConv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv3 = QConv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = QConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        self.fc0 = QLinear(6 * 6 * 256, 4096)
        self.fc1 = QLinear(4096, 4096)
        self.fc2 = QLinear(4096, num_classes)

        self.model_name = 'QAlexNet'
        self.input_shape = (1, 3, 224, 224)

    def forward(self, x) -> torch.Tensor:
        x = self.conv0(x)
        x = self.relu(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = QAlexNet()
    model.execute(is_random_input=True, fix_random_seed=True,
                  result_path='temp/QAlexNet/o_0_0_0.dat', export_onnx_path='temp/QAlexNet/QAlexNet.onnx')
