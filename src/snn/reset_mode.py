# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from enum import Enum


class ResetMode(Enum):
    '''脉冲神经元的膜电位的复位模式

    HARD: 膜电位复位到固定值
    SOFT: 将膜电位减去一个变量
    SOFT_CONSTANT: 将膜电位减去一个固定值
    '''
    HARD = 0
    SOFT = 1
    SOFT_CONSTANT = 2