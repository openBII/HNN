# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch


class Sampler(torch.nn.Module):
    '''采样器

    抽象类, 一般用于ANN到SNN的转换使用, 将静态的数据采样成时间序列

    Args:
        window_size: 采样后的时间序列的长度
    '''
    def __init__(self, window_size) -> None:
        super(Sampler, self).__init__()
        self.window_size = window_size