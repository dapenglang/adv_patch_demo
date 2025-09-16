# main_batch.py
# 文档说明：此文件用于执行批量对抗补丁攻击的主程序
# 撰写人：Bruce lang
# 日期：2024-07-15
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
import os, random

from models import load_model
from patch_attack import generate_patch
from imagenet_labels import load_imagenet_labels


def main():
    # 1. 硬件选择
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载模型
    model, preprocess = load_model("resnet18", device)
    class_names = load_imagenet_labels()
    num_classes = len(class_names)  # 1000

    # 3. CIFAR-10 数据集 (取200张)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    subset = Subset(dataset, range(200))
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    # 4. 输出目录
    os.makedirs("adv_examples", exist_ok=True)

    # 统计
    correct_orig = 0
    attack_success = 0

    # 5. 遍历图片
    for idx, (img, label) in enumerate(loader):
        img = img.to(device)

        # 掩码：左上角56x56
        mask = torch.zeros((1,1,224,224)).to(device)
        mask[:,:,0:56,0:56] = 1.0

        # 原始预测
        with torch.no_grad():
            pred_orig = model(img).argmax(1).item()
        if pred_orig == label.item():
            correct_orig += 1

        # 随机目标类别，避免等于原始预测类
        target_class = random.randint(0, num_classes - 1)
        while target_class == pred_orig:
            target_class = random.randint(0, num_classes - 1)

        # 生成对抗样本
        patched_img = generate_patch(
            model, img, target_class, mask,
            steps=30, lr=0.05, device=device
        )

        # 攻击后预测
        with torch.no_grad():
            pred_patched = model(patched_img).argmax(1).item()

        # 统计成功率（攻击后预测 = 目标类别）
        if pred_patched == target_class:
            attack_success += 1

        # 保存文件：包含原始预测 + 目标类 + 攻击后预测
        save_path = (
            f"adv_examples/img_{idx:03d}_orig_{pred_orig}_"
            f"to_{target_class}_pred_{pred_patched}.png"
        )
        utils.save_image(patched_img.cpu(), save_path)

        if (idx+1) % 20 == 0:
            print(f"Processed {idx+1}/200 images")

    # 6. 打印统计结果
    print(f"Baseline accuracy on 200 images: {correct_orig/200:.2%}")
    print(f"Attack success rate (random target classes): {attack_success/200:.2%}")


if __name__ == "__main__":
    main()
