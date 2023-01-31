# 加载ResNet50预训练模型并进行量化感知训练

1. 加载预训练模型, 测试替换后的模型是否一致 `python train_resnet50.py --pretrain --eval --test_batch_size=256 --env_gpu=0`
2. BN融合 `python train_resnet50.py --pretrain --fuse_bn --checkpoint --env_gpu=0`
3. 约束训练 `python train_resnet50.py --pretrain --train --checkpoint --restrict --lr=1e-5 -b64 --test_batch_size=256 --env_gpu=0`
4. 后训练静态量化 `python train_resnet50.py --pretrain --collect --quantize --eval --checkpoint`, checkpoint包括模型的state dict和量化参数的字典(这个步骤可以跳过)
5. 量化感知训练 `python train_resnet.py --pretrain --collect --aware --train --checkpoint --lr=1e-5 -b64 --test_batch_size=256 --env_gpu=0`
6. 测试保存的量化模型 `python train_resnet50.py --eval --quantized_pretrain --test_batch_size=256 --env_gpu=0`