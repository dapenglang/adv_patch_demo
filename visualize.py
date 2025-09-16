# visualize.py
import matplotlib.pyplot as plt

def show_results(original, patched, pred_orig, pred_patched, class_names=None):
    """
    展示原图 vs 对抗图
    """
    to_img = lambda x: x.detach().cpu().squeeze(0).permute(1,2,0).numpy()

    fig, axes = plt.subplots(1,2, figsize=(8,4))
    axes[0].imshow(to_img(original))
    title0 = f"Orig pred: {pred_orig}"
    if class_names:
        title0 += f" ({class_names[pred_orig]})"
    axes[0].set_title(title0)
    axes[0].axis("off")

    axes[1].imshow(to_img(patched))
    title1 = f"Patched pred: {pred_patched}"
    if class_names:
        title1 += f" ({class_names[pred_patched]})"
    axes[1].set_title(title1)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
