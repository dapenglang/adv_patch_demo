# patch_attack.py
import torch
import torch.nn.functional as F

def generate_patch(model, img, target_class, mask, steps=50, lr=0.01, device="cpu"):
    """
    针对单张图片生成补丁
    img: 1x3x224x224
    target_class: int
    mask: 1x1x224x224 (补丁区域=1)
    """
    img, mask = img.to(device), mask.to(device)
    model.eval()

    patch = torch.rand_like(img, requires_grad=True)  # 随机初始化补丁
    optimizer = torch.optim.Adam([patch], lr=lr)

    for _ in range(steps):
        patched = img * (1 - mask) + patch * mask
        logits = model(patched)
        loss = F.cross_entropy(logits, torch.tensor([target_class], device=device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            patch.clamp_(0, 1)

    patched_img = img * (1 - mask) + patch.detach() * mask
    return patched_img
