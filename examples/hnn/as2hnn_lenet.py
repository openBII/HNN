# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import torch.nn as nn

from hnn.ann.q_conv2d import QConv2d
from hnn.hu.a2s_poisson_coding_sign_convert import A2SPoissonCodingSignConvert
from hnn.snn.lif import QLIF
from hnn.snn.output_rate_coding import OutputRateCoding
from hnn.snn.q_linear import QLinear
from hnn.snn.q_model import QModel


class HNNLeNet(QModel):
    def __init__(self, T):
        super(HNNLeNet, self).__init__(time_window_size=T)
        self.conv1 = QConv2d(in_channels=1, out_channels=6,
                             kernel_size=5, stride=1, padding=2, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #  Nx6x16x16
        self.conv2 = QConv2d(in_channels=6, out_channels=16,
                             kernel_size=5, stride=1, padding=0, bias=False)
        #  Nx16x12x12
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #  Nx16x6x6
        self.a2shu = A2SPoissonCodingSignConvert(
            window_size=T, non_linear=torch.nn.ReLU())
        self.linear1 = QLinear(in_features=400, out_features=120)
        self.lif1 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        self.linear2 = QLinear(in_features=120, out_features=84)
        self.lif2 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        self.linear3 = QLinear(in_features=84, out_features=10)
        self.lif3 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)
        self.coding = OutputRateCoding()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        # A2SHU
        x = self.a2shu(x)  # [N, C, H, W] -> [N, C, H, W, T]
        spike = torch.zeros((self.T, 10, 10))
        x = x.permute(4, 0, 1, 2, 3)  # [T, N, C, H, W]
        input = x.view(x.size(0), x.size(1), -1)  # [T, N, C * H * W]
        v1 = None
        v2 = None
        v3 = None
        for i in range(self.T):
            x, q = self.linear1(input[i])
            out, v1 = self.lif1(x, q, v1)

            x, q = self.linear2(out)
            out, v2 = self.lif2(x, q, v2)

            x, q = self.linear3(out)
            out, v3 = self.lif3(x, q, v3)

            spike[i] = out
        return self.coding(spike)


if __name__ == '__main__':
    x = torch.randn((10, 1, 28, 28))
    model = HNNLeNet(5)
    y = model(x)
    print(y.shape)

    torch.onnx.export(model, x, 'temp/a2s_hnn_lenet.onnx',
                      custom_opsets={'snn': 1}, opset_version=11)
