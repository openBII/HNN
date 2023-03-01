# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch

from hnn.snn.lif import QLIF
from hnn.snn.output_rate_coding import OutputRateCoding
from hnn.snn.q_linear import QLinear
from hnn.snn.q_model import QModel


class SNNMLP(QModel):
    def __init__(self, in_channels, T, num_classes=10):
        super(SNNMLP, self).__init__(time_window_size=T)
        self.linear1 = QLinear(
            in_features=in_channels, out_features=num_classes, bias=False, is_encoder=True)
        self.lif1 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)

        self.linear2 = QLinear(
            in_features=num_classes, out_features=num_classes, bias=False)
        self.lif2 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)

        self.linear3 = QLinear(
            in_features=num_classes, out_features=num_classes, bias=False)
        self.lif3 = QLIF(v_th=1, v_leaky_alpha=0.9,
                         v_leaky_beta=0, v_reset=0)

        self.coding = OutputRateCoding()

    def forward(self, x: torch.Tensor):
        spike = torch.zeros((self.T, x.shape[0], x.shape[1]))
        v1 = None
        v2 = None
        v3 = None
        for i in range(self.T):
            x, q = self.linear1(x)
            out, v1 = self.lif1(x, q, v1)

            x, q = self.linear2(out)
            out, v2 = self.lif2(x, q, v2)

            x, q = self.linear3(out)
            out, v3 = self.lif3(x, q, v3)
            spike[i] = out
        return self.coding(spike)


if __name__ == '__main__':
    x = torch.randn((2, 10))
    model = SNNMLP(10, 10, 10)
    y = model(x)

    torch.onnx.export(model, x, 'temp/SNNMLP.onnx',
                      custom_opsets={'snn': 1}, opset_version=11)
