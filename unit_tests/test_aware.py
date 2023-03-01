# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import unittest
import torch
import torch.nn as nn

from hnn.ann.q_conv2d import QConv2d
from hnn.ann.q_linear import QLinear
from hnn.ann.q_model import QModel


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


class TestAware(unittest.TestCase):
    def test_aware(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LeNet()
        qmodel = QLeNet()
        qmodel.load_state_dict(model.state_dict())
        model.to(device)
        qmodel.to(device)
        qmodel.collect_q_params()
        qmodel.quantize()
        x = torch.randn((2, 1, 28, 28))
        qx = (x / x.abs().max()).mul(128).round().clamp(-128, 127)
        qx = qx.to(device)
        qy = qmodel(qx)
        x = (x / x.abs().max()).mul(128).round().clamp(-128, 127).div(128)
        qmodel.aware()
        x = x.to(device)
        y = qmodel(x)
        self.assertTrue((qy / y).equal((qy / y).mean() * torch.ones_like(y)))

if __name__ == '__main__':
    t1 = TestAware()
    t1.test_aware()
