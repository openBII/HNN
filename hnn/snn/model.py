import torch
from enum import Enum


class InputMode(Enum):
    STATIC = 'static'
    SEQUENTIAL = 'sequential'


class Model(torch.nn.Module):
    def __init__(self, time_interval: int, mode: InputMode) -> None:
        super(Model, self).__init__()
        self.time_interval = time_interval
        self.mode = mode

    def multi_step_forward(self, x, *args):
        outputs = []
        if self.mode == InputMode.STATIC:
            for i in range(self.time_interval):
                output, *args = self.forward(x, *args)
                outputs.append(output)
        elif self.mode == InputMode.SEQUENTIAL:
            for i in range(self.time_interval):
                output, *args = self.forward(x[i], *args)
                outputs.append(output)
        else:
            raise ValueError('Unsupported input mode')
        return outputs
