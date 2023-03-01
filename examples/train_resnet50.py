# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import os
import time
from typing import Any, Callable, List, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor

from hnn.ann import QAdaptiveAvgPool2d, QAdd, QConv2d, QLinear, QModel
from hnn.fuse_bn import fuse_modules

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias=False) \
        -> QConv2d:
    """3x3 convolution with padding"""
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias=False) -> QConv2d:
    """1x1 convolution"""
    return QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            batch_norm: bool = True,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.batch_norm = batch_norm
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(
            inplanes, width, stride=stride, bias=not batch_norm)
        self.relu1 = nn.ReLU(inplace=True)

        if self.batch_norm:
            self.bn1 = norm_layer(width)
            self.bn2 = norm_layer(width)
            self.bn3 = norm_layer(planes * self.expansion)
        self.conv2 = conv3x3(width, width, 1, groups,
                             dilation, bias=not batch_norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(
            width, planes * self.expansion, bias=not batch_norm)

        self.downsample = downsample

        self.add = QAdd()

        self.relu3 = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu3(out)

        return out


class ResNet(QModel):

    def __init__(
            self,
            block: Type[Bottleneck],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            batch_norm: bool = True,
    ) -> None:
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.batch_norm = batch_norm
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = QConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=(3, 3),
                             bias=not batch_norm)
        if self.batch_norm:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = QAdaptiveAvgPool2d((1, 1), 7)
        self.fc = QLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, QConv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block: Type[Bottleneck], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.batch_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion,
                            stride, not self.batch_norm),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion,
                            stride, not self.batch_norm)
                )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer, batch_norm=self.batch_norm)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, batch_norm=self.batch_norm))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        block: Type[Bottleneck],
        layers: List[int],
        batch_norm: bool = True,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, batch_norm=batch_norm, **kwargs)
    return model


def resnet50(batch_norm: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], batch_norm=batch_norm, **kwargs)


if __name__ == '__main__':
    import argparse

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--env_gpu', default='0', type=str)
    parser.add_argument('-j', '--num_workers', default=7, type=int,
                        metavar='N',
                        help='how many subprocesses to use for data loading')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--collect', dest='collect', action='store_true',
                        help='Collect quantization parameters or not')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--test_batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='wd')
    parser.add_argument('-e', '--eval', dest='eval', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Train model on training set')
    parser.add_argument('--pretrain', dest='pretrain', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--quantized_pretrain', dest='quantized_pretrain', action='store_true',
                        help='use quantized pre-trained model')
    parser.add_argument('--fuse_bn', dest='fuse_bn', action='store_true',
                        help='Fuse BN or not')
    parser.add_argument('--restrict', dest='restrict', action='store_true',
                        help='Clamp activations or not')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                        help='Save checkpoints or not')
    parser.add_argument('--quantize', dest='quantize', action='store_true',
                        help='Inference with fixed-point model')
    parser.add_argument('--aware', dest='aware', action='store_true',
                        help='Quantization-aware training')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.env_gpu
    print('Using GPU', args.env_gpu)
    print('Batch size: {:d}'.format(args.batch_size))
    print('Test batch size: {:d}'.format(args.test_batch_size))
    print('Learning rate: {:.6f}'.format(args.lr))

    c_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

    # 数据集
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        '/home/qhy/data/imagenet/train',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    test_dataset = datasets.ImageFolder(
        '/home/qhy/data/imagenet/val',
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = resnet50(batch_norm=args.fuse_bn)

    device = torch.device(
        'cuda:' + args.env_gpu if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.pretrain:
        model.load_model(
            model_path='./checkpoints/resnet50_fuse_bn_test.pth',
            device=device
        )

    if args.fuse_bn:
        model_fused = fuse_modules(model, [
            ['conv1', 'bn1'],
            ['layer1.0.conv1', 'layer1.0.bn1'],
            ['layer1.0.conv2', 'layer1.0.bn2'],
            ['layer1.0.conv3', 'layer1.0.bn3'],
            ['layer1.0.downsample.0', 'layer1.0.downsample.1'],
            ['layer1.1.conv1', 'layer1.1.bn1'],
            ['layer1.1.conv2', 'layer1.1.bn2'],
            ['layer1.1.conv3', 'layer1.1.bn3'],
            ['layer1.2.conv1', 'layer1.2.bn1'],
            ['layer1.2.conv2', 'layer1.2.bn2'],
            ['layer1.2.conv3', 'layer1.2.bn3'],
            ['layer2.0.conv1', 'layer2.0.bn1'],
            ['layer2.0.conv2', 'layer2.0.bn2'],
            ['layer2.0.conv3', 'layer2.0.bn3'],
            ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
            ['layer2.1.conv1', 'layer2.1.bn1'],
            ['layer2.1.conv2', 'layer2.1.bn2'],
            ['layer2.1.conv3', 'layer2.1.bn3'],
            ['layer2.2.conv1', 'layer2.2.bn1'],
            ['layer2.2.conv2', 'layer2.2.bn2'],
            ['layer2.2.conv3', 'layer2.2.bn3'],
            ['layer2.3.conv1', 'layer2.3.bn1'],
            ['layer2.3.conv2', 'layer2.3.bn2'],
            ['layer2.3.conv3', 'layer2.3.bn3'],
            ['layer3.0.conv1', 'layer3.0.bn1'],
            ['layer3.0.conv2', 'layer3.0.bn2'],
            ['layer3.0.conv3', 'layer3.0.bn3'],
            ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
            ['layer3.1.conv1', 'layer3.1.bn1'],
            ['layer3.1.conv2', 'layer3.1.bn2'],
            ['layer3.1.conv3', 'layer3.1.bn3'],
            ['layer3.2.conv1', 'layer3.2.bn1'],
            ['layer3.2.conv2', 'layer3.2.bn2'],
            ['layer3.2.conv3', 'layer3.2.bn3'],
            ['layer3.3.conv1', 'layer3.3.bn1'],
            ['layer3.3.conv2', 'layer3.3.bn2'],
            ['layer3.3.conv3', 'layer3.3.bn3'],
            ['layer3.4.conv1', 'layer3.4.bn1'],
            ['layer3.4.conv2', 'layer3.4.bn2'],
            ['layer3.4.conv3', 'layer3.4.bn3'],
            ['layer3.5.conv1', 'layer3.5.bn1'],
            ['layer3.5.conv2', 'layer3.5.bn2'],
            ['layer3.5.conv3', 'layer3.5.bn3'],
            ['layer4.0.conv1', 'layer4.0.bn1'],
            ['layer4.0.conv2', 'layer4.0.bn2'],
            ['layer4.0.conv3', 'layer4.0.bn3'],
            ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
            ['layer4.1.conv1', 'layer4.1.bn1'],
            ['layer4.1.conv2', 'layer4.1.bn2'],
            ['layer4.1.conv3', 'layer4.1.bn3'],
            ['layer4.2.conv1', 'layer4.2.bn1'],
            ['layer4.2.conv2', 'layer4.2.bn2'],
            ['layer4.2.conv3', 'layer4.2.bn3']
        ])
        if args.checkpoint:
            torch.save(model_fused.state_dict(),
                       './checkpoints/resnet50_fuse_bn_test.pth')

    if args.quantized_pretrain:
        model.load_quantized_model(
            model_path='./checkpoints/quantized_resnet50_acc65.716_epoch4_2022-02-26-22-27.pth',
            q_params_path='./checkpoints/q_params_resnet50_acc65.716_epoch4_2022-02-26-22-27.npy',
            device=device
        )

    if args.restrict:
        model.restrict()

    if args.collect:
        model.collect_q_params()

    if args.quantize:
        model.quantize()

    if args.eval:
        test_correct = 0
        total_num = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                if args.restrict:
                    inputs.div_(inputs.max())
                    model.restrict()
                if args.quantize or args.quantized_pretrain:
                    inputs = (inputs / inputs.max()
                              ).mul(128).floor().clamp(-128, 127)
                outputs = model(inputs)
                _, idx = torch.max(outputs.data, 1)
                test_correct += torch.sum(idx == labels.data)
                total_num += inputs.size(0)

                if i % 10 == 0:
                    print('Test [{:d} / {:d}] correct: {:.3f}%'.format(i,
                          len(test_loader), 100 * test_correct / total_num))

        print('Test correct: {:.3f}%'.format(
            100 * test_correct / len(test_dataset)))

        test_acc = 100 * test_correct / len(test_dataset)

        if args.checkpoint:
            os.makedirs('./checkpoints', exist_ok=True)
            if args.quantize:
                model.save_quantized_model(
                    model_path='./checkpoints/quantized_resnet50_acc{:.3f}_{:s}.pth'.format(
                        test_acc, c_time),
                    q_params_path='./checkpoints/q_params_resnet50_acc{:.3f}_{:s}'.format(
                        test_acc, c_time)
                )
            else:
                torch.save(model.state_dict(
                ), './checkpoints/resnet50_acc{:.3f}_{:s}.pth'.format(test_acc, c_time))

    if args.train:
        best_acc = 0
        for epoch in range(args.epochs):
            sum_loss = 0
            train_correct = 0
            total_num = 0
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                if args.restrict:
                    inputs.div_(inputs.max())
                    model.restrict()
                if args.aware:
                    inputs.div_(inputs.max())
                    model.aware()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, idx = torch.max(outputs.data, 1)
                sum_loss += loss.item()
                train_correct += torch.sum(idx == labels.data)
                total_num += inputs.size(0)

                if i % 100 == 0:
                    print('Train Epoch: {:d} [{:d} / {:d}] loss: {:.3f} correct: {:.3f}%'.format(
                        epoch + 1, i, len(train_loader), loss.item(), 100 * train_correct / total_num))

            print('Train [{:d} / {:d}] loss: {:.3f} correct: {:.3f}%'.format(epoch + 1,
                  args.epochs, sum_loss / len(train_loader), 100 * train_correct / len(train_dataset)))

            test_correct = 0
            model.eval()
            with torch.no_grad():
                if args.aware:
                    model.quantize()
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if args.restrict:
                        inputs.div_(inputs.max())
                        model.restrict()
                    if args.aware:
                        inputs = (inputs / inputs.max()
                                  ).mul(128).floor().clamp(-128, 127)
                    outputs = model(inputs)
                    _, idx = torch.max(outputs.data, 1)
                    test_correct += torch.sum(idx == labels.data)

            print('Test [{:d} / {:d}] correct: {:.3f}%'.format(epoch +
                  1, args.epochs, 100 * test_correct / len(test_dataset)))

            test_acc = 100 * test_correct / len(test_dataset)
            if test_acc > best_acc:
                best_acc = test_acc
                if args.checkpoint:
                    os.makedirs('./checkpoints', exist_ok=True)
                    if args.restrict:
                        torch.save(model.state_dict(
                        ), './checkpoints/restricted_resnet50_acc{:.3f}_epoch{:d}_{:s}.pth'.format(best_acc, epoch + 1, c_time))
                    if args.aware:
                        model.save_quantized_model(
                            model_path='./checkpoints/quantized_resnet50_acc{:.3f}_epoch{:d}_{:s}.pth'.format(
                                best_acc, epoch + 1, c_time),
                            q_params_path='./checkpoints/q_params_resnet50_acc{:.3f}_epoch{:d}_{:s}.pth'
                        )
                    else:
                        torch.save(model.state_dict(
                        ), './checkpoints/resnet50_acc{:.3f}_epoch{:d}_{:s}.pth'.format(best_acc, epoch + 1, c_time))
