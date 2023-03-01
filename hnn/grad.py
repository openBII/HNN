# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import torch


class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(
            x, torch.as_tensor(-128 / scale), torch.as_tensor(127 / scale))
        x = x.mul(scale).round().clamp(-128, 127).div(scale)  # 量化反量化
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, x_min, x_max = ctx.saved_tensors
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        mask0 = torch.where(x < x_min, zeros, ones)
        mask1 = torch.where(x > x_max, zeros, ones)
        mask = mask0 * mask1
        grad = grad_output * mask
        return grad, None


class FakeQuantizeFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(
            x, torch.as_tensor(-128 / scale), torch.as_tensor(127 / scale))
        x = x.mul(scale).floor().clamp(-128, 127).div(scale)  # 量化反量化
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, x_min, x_max = ctx.saved_tensors
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        mask0 = torch.where(x < x_min, zeros, ones)
        mask1 = torch.where(x > x_max, zeros, ones)
        mask = mask0 * mask1
        grad = grad_output * mask
        return grad, None


class DifferentiableFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FakeQuantizeINT32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(
            x, torch.as_tensor(-2147483648 / scale), torch.as_tensor(2147483647 / scale))
        x = x.mul(scale).round().clamp(-2147483648,
                                       2147483647).div(scale)  # 量化反量化
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, x_min, x_max = ctx.saved_tensors
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        mask0 = torch.where(x < x_min, zeros, ones)
        mask1 = torch.where(x > x_max, zeros, ones)
        mask = mask0 * mask1
        grad = grad_output * mask
        return grad, None


class FakeQuantizeINT28(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(
            x, torch.as_tensor(-134217728 / scale), torch.as_tensor(134217727 / scale))
        x = x.mul(scale).round().clamp(-134217728,
                                       134217727).div(scale)  # 量化反量化
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, x_min, x_max = ctx.saved_tensors
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        mask0 = torch.where(x < x_min, zeros, ones)
        mask1 = torch.where(x > x_max, zeros, ones)
        mask = mask0 * mask1
        grad = grad_output * mask
        return grad, None
