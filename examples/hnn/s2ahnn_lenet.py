# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import torch.nn as nn

from hnn.ann.q_linear import QLinear
from hnn.hu.s2a_global_rate_coding import S2AGlobalRateCoding
from hnn.snn.lif import QLIF
from hnn.snn.q_conv2d import QConv2d
from hnn.snn.q_model import QModel


class S2AHNNLeNet(QModel):
    def __init__(self, T):
        super(S2AHNNLeNet, self).__init__(time_window_size=T)
        self.conv1 = QConv2d(in_channels=1, out_channels=6,
                             kernel_size=5, stride=1, padding=2, bias=False)
        self.lif1 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #  Nx6x14x14
        self.conv2 = QConv2d(in_channels=6, out_channels=16,
                             kernel_size=5, stride=1, padding=0, bias=False)
        self.lif2 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        #  Nx16x10x10
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #  Nx16x5x5
        self.s2a = S2AGlobalRateCoding(
            window_size=T, non_linear=torch.nn.ReLU())
        self.linear1 = QLinear(in_features=400, out_features=120)
        self.linear2 = QLinear(in_features=120, out_features=84)
        self.linear3 = QLinear(in_features=84, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor):
        spike = torch.zeros((self.T, 10, 16, 5, 5))
        v1 = None
        v2 = None
        for i in range(self.T):
            x, q = self.conv1(inputs[i])
            out, v1 = self.lif1(x, q, v1)
            out = self.maxpool1(out)

            x, q = self.conv2(out)
            out, v2 = self.lif2(x, q, v2)
            out = self.maxpool2(out)
            spike[i] = out

        spike = spike.permute(1, 2, 3, 4, 0)
        x = self.s2a(spike)
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x


if __name__ == '__main__':
    x = torch.randn((10, 10, 1, 28, 28))  # INPUT.SHAPE：[T, N, C, H, W]
    model = S2AHNNLeNet(10)
    y = model(x)
    print(y.shape)

    torch.onnx.export(model, x, 'temp/s2a_hnn_lenet.onnx',
                      custom_opsets={'snn': 1}, opset_version=11)
