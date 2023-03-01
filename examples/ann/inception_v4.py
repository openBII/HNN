# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/


import torch
import torch.nn as nn
import torch.nn.functional as F

from hnn.ann.q_conv2d import QConv2d
from hnn.ann.q_linear import QLinear
from hnn.ann.q_model import QModel


class Stem_v4(nn.Module):
    """
    stem block for Inception-v4
    """

    def __init__(self):
        super(Stem_v4, self).__init__()
        self.step1 = nn.Sequential(
            QConv2d(3, 32, 3, 2),
            QConv2d(32, 32, 3),
            QConv2d(32, 64, 3)
        )
        self.step2_pool = nn.MaxPool2d(3, 2, 0)
        self.step2_conv = QConv2d(64, 96, 3, 2)
        self.step3_1 = nn.Sequential(
            QConv2d(160, 64, 1),
            QConv2d(64, 96, 3)
        )
        self.step3_2 = nn.Sequential(
            QConv2d(160, 64, 1),
            QConv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=True),
            QConv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=True),
            QConv2d(64, 96, 3)
        )
        self.step4_pool = nn.MaxPool2d(3, 2, 0)
        self.step4_conv = QConv2d(192, 192, 3, 2)

    def forward(self, x):
        out = self.step1(x)
        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)
        out = torch.cat((tmp1, tmp2), 1)
        return out


class Inception_A(nn.Module):
    """
    Inception-A block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n3, b4_n1, b4_n3):
        super(Inception_A, self).__init__()
        self.branch1 = nn.Sequential(
            # 此处调用QAdaptiveAvgPool2d会报错，它不支持平均池化
            nn.AvgPool2d(3, 1, 1),
            QConv2d(in_channels, b1, 1)
        )
        self.branch2 = QConv2d(in_channels, b2, 1)
        self.branch3 = nn.Sequential(
            QConv2d(in_channels, b3_n1, 1),
            QConv2d(b3_n1, b3_n3, 3, padding=1)
        )
        self.branch4 = nn.Sequential(
            QConv2d(in_channels, b4_n1, 1),
            QConv2d(b4_n1, b4_n3, 3, padding=1),
            QConv2d(b4_n3, b4_n3, 3, padding=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Reduction_A(nn.Module):
    """
    Reduction-A block for Inception-v4
    """

    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch1 = nn.MaxPool2d(3, 2, 0)
        self.branch2 = QConv2d(in_channels, n, 3, 2)
        self.branch3 = nn.Sequential(
            QConv2d(in_channels, k, 1, 1, 0, bias=True),
            QConv2d(k, l, 3, 1, 1, bias=True),
            QConv2d(l, m, 3, 2, 0, bias=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)


class Inception_B(nn.Module):
    """
    Inception-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x7, b3_n7x1, b4_n1, b4_n1x7_1,
                 b4_n7x1_1, b4_n1x7_2, b4_n7x1_2):
        super(Inception_B, self).__init__()
        self.branch1 = nn.Sequential(
            # 此处调用QAdaptiveAvgPool2d会报错，它不支持平均池化
            nn.AvgPool2d(3, 1, 1),
            QConv2d(in_channels, b1, 1, 1, 0, bias=True)
        )
        self.branch2 = QConv2d(in_channels, b2, 1, 1, 0, bias=True)
        self.branch3 = nn.Sequential(
            QConv2d(in_channels, b3_n1, 1, 1, 0, bias=True),
            QConv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=True),
            QConv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=True)
        )
        self.branch4 = nn.Sequential(
            QConv2d(in_channels, b4_n1, 1, 1, 0, bias=True),
            QConv2d(b4_n1, b4_n1x7_1, (1, 7), (1, 1), (0, 3), bias=True),
            QConv2d(b4_n1x7_1, b4_n7x1_1, (7, 1), (1, 1), (3, 0), bias=True),
            QConv2d(b4_n7x1_1, b4_n1x7_2, (1, 7), (1, 1), (0, 3), bias=True),
            QConv2d(b4_n1x7_2, b4_n7x1_2, (7, 1), (1, 1), (3, 0), bias=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Reduction_B_v4(nn.Module):
    """
    Reduction-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n1x7, b3_n7x1, b3_n3):
        super(Reduction_B_v4, self).__init__()
        self.branch1 = nn.MaxPool2d(3, 2, 0)
        self.branch2 = nn.Sequential(
            QConv2d(in_channels, b2_n1, 1, 1, 0, bias=True),
            QConv2d(b2_n1, b2_n3, 3, 2, 0, bias=True)
        )
        self.branch3 = nn.Sequential(
            QConv2d(in_channels, b3_n1, 1, 1, 0, bias=True),
            QConv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=True),
            QConv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=True),
            QConv2d(b3_n7x1, b3_n3, 3, 2, 0, bias=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)


class Inception_C(nn.Module):
    """
    Inception-C block for Inception-v4 net
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x3_3x1, b4_n1,
                 b4_n1x3, b4_n3x1, b4_n1x3_3x1):
        super(Inception_C, self).__init__()
        self.branch1 = nn.Sequential(
            # 此处调用QAdaptiveAvgPool2d会报错，它不支持平均池化
            nn.AvgPool2d(3, 1, 1),
            QConv2d(in_channels, b1, 1, 1, 0, bias=True)
        )
        self.branch2 = QConv2d(in_channels, b2, 1, 1, 0, bias=True)
        self.branch3_1 = QConv2d(in_channels, b3_n1, 1, 1, 0, bias=True)
        self.branch3_1x3 = QConv2d(
            b3_n1, b3_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=True)
        self.branch3_3x1 = QConv2d(
            b3_n1, b3_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=True)
        self.branch4_1 = nn.Sequential(
            QConv2d(in_channels, b4_n1, 1, 1, 0, bias=True),
            QConv2d(b4_n1, b4_n1x3, (1, 3), (1, 1), (0, 1), bias=True),
            QConv2d(b4_n1x3, b4_n3x1, (3, 1), (1, 1), (1, 0), bias=True)
        )
        self.branch4_1x3 = QConv2d(
            b4_n3x1, b4_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=True)
        self.branch4_3x1 = QConv2d(
            b4_n3x1, b4_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=True)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        tmp = self.branch3_1(x)
        out3_1 = self.branch3_1x3(tmp)
        out3_2 = self.branch3_3x1(tmp)
        tmp = self.branch4_1(x)
        out4_1 = self.branch4_1x3(tmp)
        out4_2 = self.branch4_3x1(tmp)
        return torch.cat((out1, out2, out3_1, out3_2, out4_1, out4_2), 1)


class QInception(QModel):
    """
    implementation of Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2
    """

    def __init__(self, num_classes):
        super(QInception, self).__init__()
        self.stem = Stem_v4()
        self.inception_A = self.__make_inception_A()
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()
        self.fc = QLinear(1536, num_classes)
        self.averagepool = nn.AvgPool2d(7)
        # self.dropout = nn.Dropout2d(0.2, training=self.training)
        # self.softmax = nn.Softmax2d()
        self.model_name = 'QInceptionNet-v4'
        self.input_shape = (1, 3, 299, 299)

    def __make_inception_A(self):
        layers = []
        for _ in range(4):
            layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        return Reduction_A(384, 192, 224, 256, 384)  # 1024

    def __make_inception_B(self):
        layers = []
        for _ in range(7):
            layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                      192, 192, 224, 224, 256))   # 1024
        return nn.Sequential(*layers)

    def __make_reduction_B(self):
        return Reduction_B_v4(1024, 192, 192, 256, 256, 320, 320)  # 1536

    def __make_inception_C(self):
        layers = []
        for _ in range(3):
            layers.append(Inception_C(1536, 256, 256,
                          384, 256, 384, 448, 512, 256))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.inception_A(out)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.Reduction_B(out)
        out = self.inception_C(out)
        out = self.averagepool(out)
        out = F.dropout(out, 0.2, training=self.training)
        # out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.softmax(out)
        # out = self.softmax(out)
        return out


def qinception_v4(classes=1000):
    return QInception(classes)


if __name__ == '__main__':
    model = qinception_v4()
    model.execute(is_random_input=True, fix_random_seed=True,
                  result_path='temp/QInception-v4/o_0_0_0.dat', export_onnx_path='temp/QInception-v4/QInception-v4.onnx')
