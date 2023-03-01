# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import unittest
from hnn.snn.q_model import QModel
from hnn.snn.q_linear import QLinear
from hnn.snn.lif import QLIF
from spikingjelly.clock_driven import encoding


class SNN(QModel):
    def __init__(self):
        super(SNN, self).__init__()
        self.linear = QLinear(28 * 28, 10, bias=False)
        self.lif = QLIF(v_th=1, v_leaky_alpha=0.5,
                        v_leaky_beta=0, v_reset=0)

    def forward(self, x: torch.Tensor, v: torch.Tensor = None):
        x = x.view(x.size(0), -1)
        x, q_param = self.linear(x)
        out, v = self.lif(x, q_param, v)
        return out, v


class TestSNN(unittest.TestCase):
    def test_snn(self):
        x = torch.rand(1, 1, 28, 28)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        encoder = encoding.PoissonEncoder().to(device)
        snn = SNN().to(device)
        snn.collect_q_params()
        snn.quantize()
        snn.aware(x)
        length = 2
        v = None
        for _ in range(length):
            x = encoder(x)
            spike, v = snn(x, v)


if __name__ == '__main__':
    t1 = TestSNN()
    t1.test_snn()