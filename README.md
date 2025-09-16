# 1. 环境准备

确保安装了必要依赖：

```
pip install torch torchvision matplotlib
```

项目目录结构应如下：

```
adv_patch_demo/
├── main_batch.py
├── models.py
├── patch_attack.py
├── visualize.py
├── imagenet_labels.py
└── make_grid_labeled.py

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

```
img_XXX_orig_OOO_to_TTT_pred_PPP.png
```

示例：
```
img_001_orig_5_to_7_pred_7.png
```
表示：
* 原始预测类 = 207
* 目标类 = 123
* 攻击后预测类 = 281

输出结果类似：
```
Processed 20/200 images
Processed 40/200 images
...
Baseline accuracy on 200 images: 85.50%
Attack success rate (random target classes): 62.00%

```

# 3. 生成拼图可视化

运行拼图脚本：
```
python make_grid_labeled.py

```
执行过程：

* 从 adv_examples/ 读取前 100 张图像（默认 10×10 网格）。
* 每张图底部显示标签：
```
原始类 → 攻击后类 (target 目标类)
```
实例：
```
207:goldfish → 281:tabby cat (target 123:otterhound)

```
输出文件：
```
adv_grid_labeled.png
```
# 4. 可选参数
控制网格行列

例如生成 12×12 的拼图：
```
python make_grid_labeled.py --nrow 12
```
调整缩放尺寸与字体
```
python make_grid_labeled.py --cell 196 --font 12
```
不显示类别名，仅显示 ID
```
python make_grid_labeled.py --no-names
```
# 5. 实验结果

通过上述步骤，你将得到：

1. adv_examples/ 文件夹下的 200 张对抗样本；

2. adv_grid_labeled.png 拼图，展示攻击效果：

- 每张图片显示补丁后分类从原始类 → 攻击后类 (目标类) 的转移。