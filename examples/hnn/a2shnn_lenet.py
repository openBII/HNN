# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.hu.a2s_learnable_coding import A2SLearnableCoding
from hnn.snn.lif import LIF
from hnn.snn.output_rate_coding import OutputRateCoding
from hnn.snn.model import Model
from hnn.hu.model import A2SModel
from hnn.snn.model import InputMode


class ANN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6,
                                     kernel_size=5, stride=1, padding=2)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16,
                                     kernel_size=5, stride=1, padding=0)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu(x)
        return x
    

class SNN(Model):
    def __init__(self, time_interval, mode) -> None:
        super().__init__(time_interval, mode)
        self.linear1 = torch.nn.Linear(in_features=400, out_features=120)
        self.lif1 = LIF(v_th=1, v_leaky_alpha=0.5,
                        v_leaky_beta=0, v_reset=0, v_leaky_adpt_en=False)
        self.linear2 = torch.nn.Linear(in_features=120, out_features=84)
        self.lif2 = LIF(v_th=1, v_leaky_alpha=0.5,
                        v_leaky_beta=0, v_reset=0, v_leaky_adpt_en=False)
        self.linear3 = torch.nn.Linear(in_features=84, out_features=10)
        self.lif3 = LIF(v_th=1, v_leaky_alpha=0.5,
                        v_leaky_beta=0, v_reset=0, v_leaky_adpt_en=False)
        
    def forward(self, x, v1=None, v2=None, v3=None):
        x = self.linear1(x)
        x, v1 = self.lif1(x, v1)
        x = self.linear2(x)
        x, v2 = self.lif2(x, v2)
        x = self.linear3(x)
        x, v3 = self.lif3(x, v3)
        return x, v1, v2, v3


class HNN(A2SModel):
    def __init__(self, T):
        super().__init__(T=T)
        self.ann = ANN()
        self.snn = SNN(time_interval=T, mode=InputMode.SEQUENTIAL)
        self.a2shu = A2SLearnableCoding(window_size=T, converter=torch.nn.Identity(), non_linear=torch.nn.ReLU())
        self.encode = OutputRateCoding()

    def reshape(self, x: torch.Tensor):
        x = x.view(x.size(0), -1, x.size(-1))
        return x.permute(2, 0, 1)


if __name__ == '__main__':
    x = torch.randn((10, 1, 28, 28))
    model = HNN(5)
    y = model(x)
    print(y.shape)
