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


class SNNVGG16(QModel):
    def __init__(self, T=10, num_classes=1000):
        super(SNNVGG16, self).__init__(time_window_size=T)
        self.classes = num_classes
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = QConv2d(3, 64, 3, stride=1, padding=1)
        self.lif0 = QLIF(1, 0.9, 0, 0)
        self.conv1 = QConv2d(64, 64, 3, stride=1, padding=1)
        self.lif1 = QLIF(1, 0.9, 0, 0)
        self.maxpool0 = nn.MaxPool2d(2, stride=2)
        self.conv2 = QConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.lif2 = QLIF(1, 0.9, 0, 0)
        self.conv3 = QConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.lif3 = QLIF(1, 0.9, 0, 0)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv4 = QConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.lif4 = QLIF(1, 0.9, 0, 0)
        self.conv5 = QConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.lif5 = QLIF(1, 0.9, 0, 0)
        self.conv6 = QConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.lif6 = QLIF(1, 0.9, 0, 0)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv7 = QConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.lif7 = QLIF(1, 0.9, 0, 0)
        self.conv8 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.lif8 = QLIF(1, 0.9, 0, 0)
        self.conv9 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.lif9 = QLIF(1, 0.9, 0, 0)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.conv10 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.lif10 = QLIF(1, 0.9, 0, 0)
        self.conv11 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.lif11 = QLIF(1, 0.9, 0, 0)
        self.conv12 = QConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.lif12 = QLIF(1, 0.9, 0, 0)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.fc0 = QLinear(512 * 7 * 7, 4096)
        self.fc0lif0 = QLIF(1, 0.9, 0, 0)
        self.fc1 = QLinear(4096, 4096)
        self.fc1lif1 = QLIF(1, 0.9, 0, 0)
        self.fc2 = QLinear(4096, num_classes)
        self.fc2lif2 = QLIF(1, 0.9, 0, 0)
        self.coding = OutputRateCoding()

    def forward(self, x: torch.Tensor):
        spike = torch.zeros((self.T, 2, self.classes))
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v_fc0, v_fc1, v_fc2 = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        x_copy = x
        for i in range(self.T):
            x, q = self.conv0(x_copy)
            out, v0 = self.lif0(x, q, v0)
            x = self.relu(out)

            x, q = self.conv1(x)
            out, v1 = self.lif1(x, q, v1)
            x = self.relu(out)
            x = self.maxpool0(x)

            x, q = self.conv2(x)
            out, v2 = self.lif2(x, q, v2)
            x = self.relu(out)

            x, q = self.conv3(x)
            out, v3 = self.lif3(x, q, v3)
            x = self.relu(out)
            x = self.maxpool1(x)

            x, q = self.conv4(x)
            out, v4 = self.lif4(x, q, v4)
            x = self.relu(out)

            x, q = self.conv5(x)
            out, v5 = self.lif5(x, q, v5)
            x = self.relu(out)

            x, q = self.conv6(x)
            out, v6 = self.lif6(x, q, v6)
            x = self.relu(out)
            x = self.maxpool2(x)

            x, q = self.conv7(x)
            out, v7 = self.lif7(x, q, v7)
            x = self.relu(out)

            x, q = self.conv8(x)
            out, v8 = self.lif8(x, q, v8)
            x = self.relu(out)

            x, q = self.conv9(x)
            out, v9 = self.lif9(x, q, v9)
            x = self.relu(out)
            x = self.maxpool3(x)

            x, q = self.conv10(x)
            out, v10 = self.lif10(x, q, v10)
            x = self.relu(out)

            x, q = self.conv11(x)
            out, v11 = self.lif11(x, q, v11)
            x = self.relu(out)

            x, q = self.conv12(x)
            out, v12 = self.lif12(x, q, v12)
            x = self.relu(out)
            x = self.maxpool4(x)

            x = torch.flatten(x, 1)
            x, q = self.fc0(x)
            out, v_fc0 = self.fc0lif0(x, q, v_fc0)
            x = self.relu(out)

            x, q = self.fc1(x)
            out, v_fc1 = self.fc1lif1(x, q, v_fc1)
            x = self.relu(out)

            x, q = self.fc2(x)
            out, v_fc2 = self.fc2lif2(x, q, v_fc2)
            spike[i] = out
        return self.coding(spike)


if __name__ == '__main__':
    model = SNNVGG16(T=10)
    x = torch.randn([2, 3, 224, 224])
    y = model(x)

    torch.onnx.export(model, x, 'temp/SNNVGG16.onnx',
                      custom_opsets={'snn': 1}, opset_version=11)
