# 1. 环境准备

确保安装了必要依赖：

```
pip install torch torchvision matplotlib
```

项目目录结构应如下：

```
adv_patch_demo/
├── patch_attack.py
├── visualize.py
├── imagenet_labels.py
├── README.md
```

# 2. 生成对抗样本

运行批量攻击脚本：

```
python patch_attack.py
```
执行过程：
* 自动下载 CIFAR-10 数据集（200 张）。
* 使用 ResNet18 (ImageNet预训练) 模型进行分类。
* 每张图随机选择一个目标类别（避免与原始预测类相同）。
* 在左上角生成 56x56 的对抗补丁。
* 保存对抗样本到 adv_examples/ 目录，文件名格式：

