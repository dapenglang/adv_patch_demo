# models.py
import torch
from torchvision import models, transforms

def load_model(name="resnet18", device="cpu"):
    if name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif name == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unknown model {name}")
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return model, preprocess
