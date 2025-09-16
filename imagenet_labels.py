# imagenet_labels.py
import os
import urllib.request

IMAGENET_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

def load_imagenet_labels():
    """
    下载并加载 ImageNet 1000 类标签
    """
    local_file = "imagenet_classes.txt"
    if not os.path.exists(local_file):
        print("Downloading ImageNet class labels...")
        urllib.request.urlretrieve(IMAGENET_URL, local_file)

    with open(local_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
