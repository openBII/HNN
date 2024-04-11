import torch
import torch.nn as nn
import torch.nn.functional as F

from hnn.ann.q_conv2d import QConv2d
from hnn.ann.q_model import QModel as QModel
from hnn.hu.a2s_poisson_coding_sign_convert import A2SPoissonCodingSignConvert
from hnn.snn.accumulate import Accumulate
from hnn.snn.leaky import Leaky
from hnn.snn.reset_after_spike import ResetAfterSpike
from hnn.snn.reset_mode import ResetMode
from hnn.snn.fire import Fire
from hnn.snn.surrogate.rectangle import Rectangle
from hnn.snn.q_conv2d import QConv2d as SNNQConv2d
from hnn.snn.q_model import QModel as SNNQModel


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, last_relu=False, downsample=None, stride2=False):
        super(Bottleneck, self).__init__()
        self.conv1 = QConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=2 if stride2 else 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.last_relu = last_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x) if self.downsample is not None else x
        out += residual
        out = out[:, :, 1:-1, 1:-1].contiguous()  # 不要最外面一圈
        return out if not self.last_relu else self.relu(out)


class ResNet2Stage(QModel):
    def __init__(self, firstchannels=64, channels=(64, 128), inchannel=3, block_num=(3, 4)):
        self.inplanes = firstchannels
        super(ResNet2Stage, self).__init__()
        self.conv1 = QConv2d(inchannel, firstchannels, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = self._make_layer(channels[0], block_num[0], last_relu=True, stride2=True)
        self.stage2 = self._make_layer(channels[1], block_num[1], last_relu=True, stride2=True)
        self.conv_out = QConv2d(channels[1] * 4, channels[1] * 4, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, QConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, planes, blocks, last_relu, stride2=False):
        block = Bottleneck
        downsample = None
        if self.inplanes != planes * block.expansion or stride2:
            downsample = nn.Sequential(
                QConv2d(self.inplanes, planes * block.expansion, kernel_size=3,
                        stride=2 if stride2 else 1, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, last_relu=True, downsample=downsample, stride2=stride2)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, last_relu=(last_relu if i == (blocks-1) else True)))

        return nn.Sequential(*layers)

    def step(self, x):
        x = self.conv1(x)  # stride = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.stage1(x)  # stride = 4
        x = self.stage2(x)  # stride = 8
        x = self.conv_out(x)
        return x

    def forward(self, net_in):
        return torch.stack([self.step(net_in[..., step]) for step in range(net_in.shape[-1])], -1)


class SNNLayer(nn.Module):
    def __init__(self, layer: SNNQConv2d, bn=True, thresh=None, thresh_grad=True, decay=0.0, decay_grad=False, bypass_in=False, update_v='default'):
        super(SNNLayer, self).__init__()
        self.layer = layer
        self.state = [0., 0.]  # [mem, spike]
        self.hard_reset = ResetAfterSpike(reset_mode=ResetMode.HARD, v_reset=0)
        self.soft_reset = ResetAfterSpike(reset_mode=ResetMode.SOFT)
        self.leaky = Leaky(alpha=decay, beta=0)
        self.accumulate = Accumulate(v_init=0)
        self.fire = Fire(surrogate_function=Rectangle)

        if thresh is None:
            thresh = 0.5

        self.thresh = nn.Parameter(torch.ones((1, layer.out_channels, 1, 1)) * thresh, requires_grad=thresh_grad)
        self.decay = nn.Parameter(torch.ones((1, layer.out_channels, 1, 1)) * decay, requires_grad=decay_grad)

        self.bn = nn.BatchNorm2d(layer.out_channels) if bn else None
        self.bypass_bn = nn.BatchNorm2d(layer.out_channels) if bn and bypass_in else None

        if bn and thresh:
            self.bn.weight.data = self.thresh.data.view(-1) / (2 ** 0.5 if bypass_in else 1)
            if bypass_in:
                self.bypass_bn.weight.data = self.thresh.data.view(-1) / 2 ** 0.5

        self.update_v = update_v

    def update_state(self, x, bypass_in):
        layer_in = self.bn(self.layer(x)) if self.bn is not None else self.layer(x)
        if bypass_in is not None:
            layer_in += self.bypass_bn(bypass_in) if self.bn is not None else bypass_in

        if self.update_v == 'default':
            self.state[0] = self.hard_reset(self.state[0], self.state[1])
            self.state[0] = self.leaky(self.state[0])
            self.state[0] = self.accumulate(layer_in, self.state[0])
        elif self.update_v == 'bursting':
            self.state[0] = self.leaky(self.state[0])
            self.state[0] = self.soft_reset(self.state[0], self.state[1], self.thresh)
            self.state[0] = self.accumulate(layer_in, self.state[0])
        elif self.update_v == 'rnn':
            self.state[0] = self.leaky(self.state[0])
            self.state[0] = self.accumulate(layer_in, self.state[0])

        self.state[1] = self.fire(self.state[0], self.thresh)

    def reset_state(self, history):
        self.state = [self.state[0].detach(), self.state[1].detach()] if history else [0., 0.]
        self.decay.data = self.decay.clamp(min=0., max=1.).data

    def forward(self, x, bypass_in=None):
        self.update_state(x, bypass_in)
        return self.state[1]


class BottleneckSNN(nn.Module):
    def __init__(self, inplanes, planes, downsample=None, stride2=False, expansion=4):
        super().__init__()

        self.layers = nn.Sequential(
            SNNLayer(SNNQConv2d(inplanes, planes, kernel_size=1, bias=False)),
            SNNLayer(SNNQConv2d(planes, planes, kernel_size=3, stride=2 if stride2 else 1, padding=1, bias=False))
        )
        self.residual_layer = SNNLayer(SNNQConv2d(planes, planes * expansion, kernel_size=1, bias=False), bypass_in=True)

        self.downsample = downsample

    def reset_state(self, history):
        for layer in self.layers:
            layer.reset_state(history)
        self.residual_layer.reset_state(history)

    def forward(self, x):
        out = self.layers(x)
        residual = self.downsample(x) if self.downsample is not None else x

        out = self.residual_layer(out, residual)
        out = out[:, :, 1:-1, 1:-1].contiguous()
        return out


class ResNet2StageSNN(SNNQModel):
    expansion = 4

    def __init__(self, firstchannels=64, channels=(64, 128), inchannel=3, block_num=(3, 4)):
        super().__init__()

        self.layers = nn.Sequential(
            SNNLayer(SNNQConv2d(inchannel, firstchannels, kernel_size=7, stride=2, padding=1, bias=False)),
            *self._make_layer(firstchannels, channels[0], block_num[0], stride2=True),
            *self._make_layer(channels[0] * ResNet2StageSNN.expansion, channels[1], block_num[1], stride2=True),
            SNNLayer(SNNQConv2d(channels[1] * ResNet2StageSNN.expansion, channels[1] * ResNet2StageSNN.expansion, kernel_size=1, bias=False),
                     decay=0., decay_grad=False, update_v='rnn', bn=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, inplanes, planes, blocks, stride2=False):

        downsample = SNNQConv2d(inplanes, planes * ResNet2StageSNN.expansion, kernel_size=3, stride=2 if stride2 else 1, padding=1, bias=False)

        layers = [BottleneckSNN(inplanes, planes, downsample=downsample, stride2=stride2)]
        layers += [BottleneckSNN(planes * ResNet2StageSNN.expansion, planes) for i in range(1, blocks)]

        return layers

    def reset_state(self, history=False):
        for layer in self.layers:
            layer.reset_state(history)

    def step(self, x):
        self.layers(x)
        out = self.layers[-1].state[0]
        return out

    def forward(self, net_in):
        self.reset_state()
        out_list = []
        for t in range(net_in.shape[-1]):
            net_out = self.step(net_in[..., t])
            out_list.append(net_out)
        return torch.stack(out_list, -1)
    

class TurningDiskSiamFC(QModel):
    def __init__(self):
        super().__init__()
        self.aps_net = ResNet2Stage(inchannel=1, block_num=[1, 1])
        self.dvs_net = ResNet2StageSNN(inchannel=2, block_num=[1, 1])

    def corr_up(self, x, k):
        c = F.conv2d(x, k).unflatten(1, (x.shape[0], k.shape[0]//x.shape[0])).diagonal().permute(3, 0, 1, 2)
        return c

    @staticmethod
    def extract_clip(ff, clip_loc, clip_size):
        bs, fs, h, w = ff.shape
        ch, cw = clip_size

        tenHorizontal = torch.linspace(-1.0, 1.0, cw).expand(1, 1, ch, cw) * cw / w
        tenVertical = torch.linspace(-1.0, 1.0, ch).unsqueeze(-1).expand(1, 1, ch, cw) * ch / h
        tenGrid = torch.cat([tenHorizontal, tenVertical], 1).to(ff.device)

        clip_loc[..., 0] /= w / 2
        clip_loc[..., 1] /= h / 2
        tenDis = clip_loc.unsqueeze(-1).unsqueeze(-1).type(torch.float32)

        tenGrid = (tenGrid.unsqueeze(1) + tenDis).permute(1, 0, 3, 4, 2)
        target_list = [F.grid_sample(input=ff, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=True) for grid in tenGrid]

        return torch.stack(target_list, 1).flatten(end_dim=1)

    @staticmethod
    def gen_gt_cm(target_loc, map_size):
        w, h = map_size

        tenHorizontal = torch.arange(0, w).expand(1, 1, 1, h, w) - w / 2 + 0.5
        tenVertical = torch.arange(0, h).unsqueeze(-1).expand(1, 1, 1, h, w) - h / 2 + 0.5
        tenGrid = torch.stack([tenHorizontal, tenVertical], 2).to(target_loc.device)

        target_loc = target_loc.unsqueeze(-1).unsqueeze(-1).type(torch.float32)
        dist = torch.norm(tenGrid - target_loc, dim=2)
        gt_cm = -1 + (dist < 2) * 1 + (dist < 1) * 1
        return gt_cm.permute(0, 1, 3, 4, 2)

    def get_target_loc(self, cm, img_size):
        iw, ih = img_size
        bs, ns, h, w, ts = cm.shape
        tenHorizontal = (torch.arange(0, w).expand(1, 1, 1, h, w) - w / 2 + 0.5) * 8 + iw / 2
        tenVertical = (torch.arange(0, h).unsqueeze(-1).expand(1, 1, 1, h, w) - h / 2 + 0.5) * 8 + ih / 2
        tenGrid = torch.stack([tenHorizontal, tenVertical], 2).to(cm.device).expand(bs, ns, 2, ts, h, w)
        index = cm.permute(0, 1, 4, 2, 3).flatten(start_dim=-2).argmax(dim=-1, keepdim=True).unsqueeze(2).expand(bs, ns, 2, ts, 1)
        target_loc = tenGrid.flatten(start_dim=-2).gather(dim=-1, index=index).squeeze(dim=-1)
        return target_loc

    def forward(self, aps, dvs, aps_loc, dvs_loc, training=True):
        bs, _, h, w, ts = dvs.shape
        aps_feature = self.aps_net.step(aps)
        dvs_feature = self.dvs_net(dvs)
        kernel = self.extract_clip(aps_feature, aps_loc, (3, 3))
        cm = torch.stack([self.corr_up(dvs_feature[..., t], kernel) for t in range(ts)], -1)

        if training:
            l_reg = 0.1
            gt_cm = self.gen_gt_cm(dvs_loc, cm.shape[2:4])
            loss = - (gt_cm * cm).sum(dim=(2, 3)) + l_reg * torch.pow(cm * (gt_cm != 0), 2).sum(dim=(2, 3))
            loss = loss.mean(dim=(1, 2))
            return {"loss": loss, "cm": cm}
        else:
            pred_loc = self.get_target_loc(cm, aps.shape[-2:][::-1])
            dvs_loc = dvs_loc * 8 + torch.tensor(aps.shape[-2:][::-1]).view(1, 1, 2, 1).to(dvs_loc.device) / 2 - 0.5
            return {"cm": cm, "pred_loc": pred_loc, "aps": aps, "dvs": dvs, "gt_loc": dvs_loc}


if __name__ == "__main__":
    ann = ResNet2Stage()
    print(ann)
    snn = ResNet2StageSNN()
    print(snn)