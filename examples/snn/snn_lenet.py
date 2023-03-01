# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import torch.nn as nn

from hnn.snn.lif import QLIF
from hnn.snn.output_rate_coding import OutputRateCoding
from hnn.snn.q_conv2d import QConv2d
from hnn.snn.q_linear import QLinear
from hnn.snn.q_model import QModel


class SNNLeNet(QModel):
    def __init__(self, T):
        super(SNNLeNet, self).__init__(time_window_size=T)
        self.conv1 = QConv2d(in_channels=1, out_channels=6,
                             kernel_size=5, stride=1, padding=2, bias=False)
        #  Nx6x32x32
        self.lif1 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #  Nx6x16x16
        self.conv2 = QConv2d(in_channels=6, out_channels=16,
                             kernel_size=5, stride=1, padding=0, bias=False)
        #  Nx16x12x12
        self.lif2 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #  Nx16x6x6
        self.linear1 = QLinear(in_features=576, out_features=120)
        self.lif3 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        self.linear2 = QLinear(in_features=120, out_features=84)
        self.lif4 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        self.linear3 = QLinear(in_features=84, out_features=10)
        self.lif5 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        self.coding = OutputRateCoding()

    def forward(self, x: torch.Tensor):
        # x_seq = x.unsqueeze(0).repeat(
        #     self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T,N, C, H, W]
        # x_seq = self.snnlenet(x_seq)
        # fr = x_seq.mean(0)
        spike = torch.zeros((self.T, 10, 10))
        xx = x
        v1 = None
        v2 = None
        v3 = None
        v4 = None
        v5 = None
        for i in range(self.T):
            x, q = self.conv1(xx)
            out, v1 = self.lif1(x, q, v1)
            out = self.maxpool1(out)

            x, q = self.conv2(out)
            out, v2 = self.lif2(x, q, v2)
            out = self.maxpool2(out)

            out = torch.flatten(out, 1, -1)

            x, q = self.linear1(out)
            out, v3 = self.lif3(x, q, v3)

            x, q = self.linear2(out)
            out, v4 = self.lif2(x, q, v4)

            x, q = self.linear3(out)
            out, v5 = self.lif3(x, q, v5)

            spike[i] = out
        return self.coding(spike)


if __name__ == '__main__':
    x = torch.randn((10, 1, 32, 32))
    model = SNNLeNet(10)
    y = model(x)

    torch.onnx.export(model, x, 'temp/SNNLeNet.onnx',
                      custom_opsets={'snn': 1}, opset_version=11)
