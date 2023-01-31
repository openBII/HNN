# Hybrid Programming Framework

## 简介

HNN编程框架基于PyTorch进行开发，提供了编写ANN、SNN、HNN模型的编程接口，同时可以支持通过此编程框架描述的ANN、SNN模型的自动化量化（HNN的自动化量化仍在开发中），可以支持后训练静态量化和量化感知训练。下面对SNN和HNN编程进行简要说明：
- SNN编程由一系列基本SNN操作组成，通过这些基本操作可以组成灵活的、功能丰富的扩展LIF神经元模型，用户也可以基于这些基本操作实现自定义的神经元模型。
- HNN编程中的HNN主要指[1]中以Hybrid Unit (HU)为转换单元来连接ANN和SNN网络的混合网络，编程框架中实现了可扩展的HU，用户可使用编程框架中提供的各种HU或自定义HU。

此框架的开发过程中考虑了与BiMap的融合，通过此编程框架描述的网络可以进一步被BiMap中的编译系统编译部署到支持的类脑计算芯片上。

HNN编程框架的详细开发及使用文档请见工程文档。

## 基本使用

安装相关依赖：
```bash
pip install -r requirement.txt
```

注：目前因为ONNX版本兼容问题，Pytorch需要使用1.11.0版本

`examples`文件夹下为通过此编程框架写出的一些ANN、SNN、HNN模型，以需要量化的SNN为例，SNN模型需要继承`src.snn`中的`QModel`类，并通过`QConv2d`, `QLinear`, `QLIF`等算子来搭建网络：
```python
from src.snn import QModel, QLinear, QLIF


class SNN(QModel):
    def __init__(self, in_channels, T, num_classes=10):
        super(SNN, self).__init__(time_window_size=T)
        self.linear = QLinear(
            in_features=in_channels, out_features=num_classes)
        self.lif = QLIF(v_th=1, v_leaky_alpha=0.9,
                        v_leaky_beta=0, v_reset=0)
```


## 参考引用

如果使用到本编程框架的HNN部分，请引用[1]：

    @article{Zhao2022,
    doi = {10.1038/s41467-022-30964-7},
    url = {https://doi.org/10.1038/s41467-022-30964-7},
    year = {2022},
    month = jun,
    publisher = {Springer Science and Business Media {LLC}},
    volume = {13},
    number = {1},
    author = {Rong Zhao and Zheyu Yang and Hao Zheng and Yujie Wu and Faqiang Liu and Zhenzhi Wu and Lukai Li and Feng Chen and Seng Song and Jun Zhu and Wenli Zhang and Haoyu Huang and Mingkun Xu and Kaifeng Sheng and Qianbo Yin and Jing Pei and Guoqi Li and Youhui Zhang and Mingguo Zhao and Luping Shi},
    title = {A framework for the general design and computation of hybrid neural networks},
    journal = {Nature Communications}
    }

本工程的SNN和HNN编程部分参考或复用了部分[SpikingJelly](https://github.com/fangwei123456/spikingjelly)的代码：

    @misc{SpikingJelly,
        title = {SpikingJelly},
        author = {Fang, Wei and Chen, Yanqi and Ding, Jianhao and Chen, Ding and Yu, Zhaofei and Zhou, Huihui and Timothée Masquelier and Tian, Yonghong and other contributors},
        year = {2020},
        howpublished = {\url{https://github.com/fangwei123456/spikingjelly}},
        note = {Accessed: YYYY-MM-DD},
    }
