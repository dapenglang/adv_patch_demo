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
