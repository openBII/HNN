# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import unittest
from src.hnn.a2s_poisson_coding_sign_convert import A2SPoissonCodingSignConvert
from src.hnn.s2a_global_rate_coding import S2AGlobalRateCoding
from src.hnn.s2a_learnable_rate_coding import S2ALearnableRateCoding


class TestHNN(unittest.TestCase):
    def test_hnn(self):
        a2s = A2SPoissonCodingSignConvert(
            window_size=5, non_linear=torch.nn.ReLU())
        x = torch.randn(3, 4, 5, 6)
        y = a2s(x)
        self.assertTrue(y.shape == torch.Size([3, 4, 5, 6, 5]))

        s2a = S2AGlobalRateCoding(window_size=9, non_linear=torch.nn.ReLU())
        x = torch.randn(3, 4, 5, 6, 9).le(0.5).to(torch.float)
        y = s2a(x)
        self.assertTrue(y.shape == torch.Size([3, 4, 5, 6, 1]))

        s2a = S2ALearnableRateCoding(
            window_size=3, num_windows=3, kernel_size=3, stride=1, padding=0)
        y = s2a(x)
        self.assertTrue(y.shape == torch.Size([3, 4, 5, 6, 3]))


if __name__ == '__main__':
    t1 = TestHNN()
    t1.test_hnn()