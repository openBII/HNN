# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch
from hnn.hu.a2s_learnable_coding import A2SLearnableCoding
from hnn.snn.lif import LIF
from hnn.snn.output_rate_coding import OutputRateCoding
from hnn.snn.model import Model
from hnn.hu.model import A2SModel
from hnn.snn.model import InputMode


class ANN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6,
                                     kernel_size=5, stride=1, padding=2)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16,
                                     kernel_size=5, stride=1, padding=0)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        return x
    

class SNN(Model):
    def __init__(self, time_interval, mode) -> None:
        super().__init__(time_interval, mode)
        self.linear1 = torch.nn.Linear(in_features=400, out_features=10)
        self.lif1 = LIF(v_th=1, v_leaky_alpha=0.5,
                        v_leaky_beta=0, v_reset=0)
        
    def forward(self, x, v1=None, v2=None, v3=None):
        x = self.linear1(x)
        x, v1 = self.lif1(x, v1)
        return x, v1
    

class HNN(A2SModel):
    def __init__(self, T):
        super().__init__(T=T)
        self.ann = ANN()
        self.snn = SNN(time_interval=T, mode=InputMode.SEQUENTIAL)
        self.a2shu = A2SLearnableCoding(window_size=T, converter=torch.nn.Identity())
        self.encode = OutputRateCoding()

    def reshape(self, x: torch.Tensor):
        x = x.view(x.size(0), -1, x.size(-1))
        return x.permute(2, 0, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
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
    parser.add_argument('--train', dest='train', action='store_true', default=True,
                        help='Train model on training set')
    parser.add_argument('--pretrain', dest='pretrain', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                        help='Save checkpoints or not')
    parser.add_argument('-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
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
    
    model = HNN(T=10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.pretrain:
        state_dict = torch.load(
            './checkpoints/restricted_lenet_acc99.170_epoch4_2022-02-24-15-04.pth', map_location=device)
        model.load_state_dict(state_dict)

    if args.eval:
        test_correct = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, idx = torch.max(outputs.data, 1)
                test_correct += torch.sum(idx == labels.data)

        print('Test correct: {:.3f}%'.format(
            100 * test_correct / len(test_dataset)))

        test_acc = 100 * test_correct / len(test_dataset)

        if args.checkpoint:
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(model.state_dict(), './checkpoints/hnn_lenet_acc{:.3f}_{:s}.pth'.format(test_acc, c_time))

    if args.train:
        best_acc = 0
        for epoch in range(args.epochs):
            sum_loss = 0
            train_correct = 0
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
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
                    torch.save(model.state_dict(), './checkpoints/hnn_lenet_acc{:.3f}_epoch{:d}_{:s}.pth'.format(best_acc, epoch + 1, c_time))
