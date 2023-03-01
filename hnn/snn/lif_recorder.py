# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch
from hnn.snn.recorder import Recorder


class LIFRecorder(Recorder):
    '''LIF神经元的Recorder

    需要重载forward方法和symbolic方法, forward方法需要传入输入和神经元的各种参数, symbolic方法构建Recorder结点并记录参数
    '''
    @staticmethod
    def forward(ctx, input, v_th, v_leaky_alpha, v_leaky_beta, v_reset, v_leaky_adpt_en, v_init, time_window_size):
        return input

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, v_th, v_leaky_alpha, v_leaky_beta, v_reset, v_leaky_adpt_en, v_init, time_window_size):
        return g.op("snn::LIFRecorder", input,
                    v_th_f=v_th, v_leaky_alpha_f=v_leaky_alpha,
                    v_leaky_beta_f=v_leaky_beta, v_reset_f=v_reset,
                    v_leaky_adpt_en_i=v_leaky_adpt_en, v_init_f=v_init,
                    time_window_size_f=time_window_size)