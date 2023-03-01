# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
import unittest
import logging
from hnn.ann.q_conv2d import QConv2d


class TestQConv2d(unittest.TestCase):
    def test_q_conv2d(self):
        x = torch.randn((1, 3, 3, 3))
        qconv = QConv2d(3, 8, 3)
        conv = torch.nn.Conv2d(3, 8, 3)
        logging.debug(conv.weight.data.abs().max())
        qconv.load_state_dict(conv.state_dict())
        qconv.collect_q_params(2)
        logging.debug(qconv.weight_scale)
        logging.debug(qconv.bit_shift)
        qx = x / x.abs().max() * 128
        qx = qx.round().clamp(-128, 127)
        qconv.quantize()
        qy = qconv(qx)
        logging.debug(qy)
        qconv.aware()
        y = qconv(x / x.abs().max())
        logging.debug(y * 128)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    t1 = TestQConv2d()
    t1.test_q_conv2d()