# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from examples.ann.lenet import QLeNet
from src.ann.q_model import QModel
from src.ann.q_module import QModule


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--collect', dest='collect', action='store_true',
                        help='Collect quantization parameters or not')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
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

    c_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size, shuffle=False)

    model = QLeNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.pretrain:
        state_dict = torch.load(
            './checkpoints/restricted_lenet_acc99.170_epoch4_2022-02-24-15-04.pth', map_location=device)
        model.load_state_dict(state_dict)

    if args.restrict:
        model.restrict()

    if args.collect:
        if not(args.pretrain):
            warnings.warn(
                'Collecting quantization parameters usually requires a pretrained model')
        model.collect_q_params()

    if args.quantize:
        model.quantize()

    if args.eval:
        test_correct = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if args.quantize:
                    inputs = (inputs / inputs.max()
                              ).mul(128).round().clamp(-128, 127)
                outputs = model(inputs)
                _, idx = torch.max(outputs.data, 1)
                test_correct += torch.sum(idx == labels.data)

        print('Test correct: {:.3f}%'.format(
            100 * test_correct / len(test_dataset)))

        test_acc = 100 * test_correct / len(test_dataset)

        if args.checkpoint:
            os.makedirs('./checkpoints', exist_ok=True)
            if args.quantize:
                torch.save(model.state_dict(
                ), './checkpoints/quantized_lenet_acc{:.3f}_{:s}.pth'.format(test_acc, c_time))
                q_params_dict = {}
                for name, module in model.named_modules():
                    if isinstance(module, QModule) and not(isinstance(module, QModel)):
                        q_params_dict[name] = module.bit_shift
                np.save(
                    './checkpoints/q_params_lenet_acc{:.3f}_{:s}.pth'.format(test_acc, c_time), q_params_dict)
            else:
                torch.save(model.state_dict(
                ), './checkpoints/lenet_acc{:.3f}_{:s}.pth'.format(test_acc, c_time))

    if args.train:
        best_acc = 0
        for epoch in range(args.epochs):
            sum_loss = 0
            train_correct = 0
            model.train()
            if args.aware:
                model.aware()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if args.restrict:
                    inputs = inputs / inputs.max()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, idx = torch.max(outputs.data, 1)
                sum_loss += loss.item()
                train_correct += torch.sum(idx == labels.data)

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
                        inputs = inputs / inputs.max()
                    if args.aware:
                        inputs = (inputs / inputs.max()
                                  ).mul(128).round().clamp(-128, 127)
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
                        ), './checkpoints/restricted_lenet_acc{:.3f}_epoch{:d}_{:s}.pth'.format(best_acc, epoch + 1, c_time))
                    if args.aware:
                        torch.save(model.state_dict(
                        ), './checkpoints/quantized_lenet_acc{:.3f}_epoch{:d}_{:s}.pth'.format(best_acc, epoch + 1, c_time))
                        q_params_dict = {}
                        for name, module in model.named_modules():
                            if isinstance(module, QModule) and not(isinstance(module, QModel)):
                                q_params_dict[name] = module.bit_shift
                        np.save('./checkpoints/q_params_lenet_acc{:.3f}_epoch{:d}_{:s}.pth'.format(
                            best_acc, epoch + 1, c_time), q_params_dict)
                    else:
                        torch.save(model.state_dict(
                        ), './checkpoints/lenet_acc{:.3f}_epoch{:d}_{:s}.pth'.format(best_acc, epoch + 1, c_time))
