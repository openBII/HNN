# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import pickle

import numpy as np
import torch.nn as nn

from src.ann.q_conv2d import QConv2d
from src.ann.q_linear import QLinear
from src.ann.q_model import QModel


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        nn.Conv2d()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(400, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class QLeNet(QModel):
    def __init__(self):
        super(QLeNet, self).__init__()
        self.conv1 = QConv2d(1, 6, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = QConv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.linear1 = QLinear(400, 120)
        self.linear2 = QLinear(120, 84)
        self.linear3 = QLinear(84, 10)
        self.relu = nn.ReLU(inplace=True)

        self.model_name = 'QLeNet'
        self.input_shape = (1, 1, 28, 28)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

    def forward(self, x, record_path=None):
        if record_path is not None:
            record_dict = {}
            record_dict.update({0: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool1(x)
            record_dict.update({8: x.squeeze(0).permute(
                1, 2, 0).detach().numpy().astype(np.int32)})
            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool2(x)
            x = x.view(x.size(0), -1)
            record_dict.update(
                {18: x.squeeze(0).detach().numpy().astype(np.int32)})
            x = self.linear1(x)
            x = self.relu(x)
            record_dict.update(
                {24: x.squeeze(0).detach().numpy().astype(np.int32)})
            x = self.linear2(x)
            x = self.relu(x)
            record_dict.update(
                {30: x.squeeze(0).detach().numpy().astype(np.int32)})
            x = self.linear3(x)
            record_dict.update(
                {34: x.view(-1).detach().numpy().astype(np.int32)})
            with open(record_path, 'wb') as f:
                pickle.dump(record_dict, f)
            return x
        else:
            return self._forward(x)


if __name__ == '__main__':
    model = QLeNet()
    model.execute(is_random_input=True, fix_random_seed=True,
                  result_path='temp/QLeNet/o_0_0_0.dat', export_onnx_path='temp/QLeNet/QLeNet.onnx')
