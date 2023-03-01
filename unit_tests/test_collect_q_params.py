# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import math
import unittest

import torch
import torch.nn as nn

from hnn.ann.q_conv2d import QConv2d
from hnn.ann.q_linear import QLinear
from hnn.ann.q_model import QModel
from hnn.ann.q_module import QModule


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
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


class TestCollectQParams(unittest.TestCase):
    def test_collect_q_params(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LeNet()
        qmodel = QLeNet()
        qmodel.load_state_dict(model.state_dict())
        model.to(device)
        qmodel.to(device)
        qmodel.collect_q_params()
        compare_dict = {}
        for name, module in qmodel.named_modules():
            if not(isinstance(module, QModel)) and isinstance(module, QModule):
                compare_dict[name] = [int(module.bit_shift)]
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                compare_dict[name].append(math.log(128 / module.weight.abs().max(), 2))
                self.assertTrue(abs(compare_dict[name][0] - compare_dict[name][1]) < 1.5)


if __name__ == '__main__':
    t1 = TestCollectQParams()
    t1.test_collect_q_params()
