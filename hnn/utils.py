# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import random

import numpy
import torch


def setup_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_int8_tensor(shape):
    assert type(shape) is tuple
    x = torch.rand(shape) * 2 - 1
    x = x.mul(128).round().clamp(-128, 127)
    return x
