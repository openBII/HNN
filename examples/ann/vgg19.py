# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import pickle

import numpy as np
import torch
import torch.nn as nn

from src.ann.q_conv2d import QConv2d
from src.ann.q_linear import QLinear
from src.ann.q_model import QModel


class QVGG19(QModel):
    def __init__(self, num_classes=1000):
        super(QVGG19, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = QConv2d(3, 64, 3, stride=1, padding=1)
        self.conv1 = QConv2d(64, 64, 3, stride=1, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, stride=2)
        self.conv2 = QConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = QConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv4 = QConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = QConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = QConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = QConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv8 = QConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.conv12 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv14 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv15 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.fc0 = QLinear(512 * 7 * 7, 4096)
        self.fc1 = QLinear(4096, 4096)
        self.fc2 = QLinear(4096, num_classes)

        self.model_name = 'QVGG19'
        self.input_shape = (1, 3, 224, 224)

    def _forward(self, x) -> torch.Tensor:
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool0(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.relu(x)
        x = self.conv11(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.relu(x)
        x = self.conv14(x)
        x = self.relu(x)
        x = self.conv15(x)
        x = self.relu(x)
        x = self.maxpool4(x)

        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, x, record_path=None):
        if record_path is not None:
            record_dict = {}
            record_dict.update({0: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self._forward(x)
            record_dict.update(
                {124: x.view(-1).detach().numpy().astype(np.int32)})

            with open(record_path, 'wb') as f:
                pickle.dump(record_dict, f)
            return x
        else:
            return self._forward(x)


if __name__ == '__main__':
    model = QVGG19()
    model.execute(is_random_input=True, fix_random_seed=True,
                  result_path='temp/QVGG19/o_0_0_0.dat', export_onnx_path='temp/QVGG19/QVGG19.onnx')
