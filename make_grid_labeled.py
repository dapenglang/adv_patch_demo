# make_grid_labeled.py
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from imagenet_labels import load_imagenet_labels

def parse_fname(fname: str):
    """
    解析文件名: img_XXX_orig_OOO_to_TTT_pred_PPP.png
    返回: (orig, pred, target)
    """
    try:
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        orig = int(parts[3])
        target = int(parts[5])
        pred = int(parts[7])
        return orig, pred, target
    except Exception:
        return -1, -1, -1

def label_text(orig, pred, target, names=None, use_names=True):
    def fmt(k):
        if k < 0:
            return "?"
        if use_names and names and 0 <= k < len(names):
            return f"{k}:{names[k]}"
        return str(k)
    return f"{fmt(orig)} → {fmt(pred)} (target {fmt(target)})"

def load_font(font_size: int):
    candidates = [
        "Arial.ttf", "arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()

def make_grid_with_labels(
    folder: str = "adv_examples",
    save_path: str = "adv_grid_labeled.png",
    nrow: int = 10,
    cell_size: int = 224,
    font_size: int = 14,
    use_names: bool = True,
):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
    if not files:
        print("⚠️ 未找到对抗样本！请先运行 main_batch.py")
        return

    files = files[: nrow * nrow]
    names = load_imagenet_labels() if use_names else None

    grid_w = nrow * cell_size
    ncol = nrow
    nrows = (len(files) + ncol - 1) // ncol
    grid_h = nrows * cell_size
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    font = load_font(font_size)
    draw = ImageDraw.Draw(canvas)

    resize = transforms.Compose([transforms.Resize((cell_size, cell_size))])

    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")
        img_resized = resize(img)

        row, col = divmod(i, ncol)
        x = col * cell_size
        y = row * cell_size
        canvas.paste(img_resized, (x, y))

        orig, pred, target = parse_fname(fname)
        txt = label_text(orig, pred, target, names=names, use_names=use_names)

        tw, th = draw.textlength(txt, font=font), font_size
        rect_h = th + 6
        rect_y0 = y + cell_size - rect_h
        rect_y1 = y + cell_size
        draw.rectangle([(x, rect_y0), (x + cell_size, rect_y1)], fill=(255, 255, 255))
        draw.text((x + 4, rect_y0 + 3), txt, fill="black", font=font)

    canvas.save(save_path)
    print(f"✅ 拼图已保存到 {save_path}")

def main():
    ap = argparse.ArgumentParser(description="生成带标签的对抗样本拼图：原始类 → 攻击后类 (目标类)")
    ap.add_argument("--folder", type=str, default="adv_examples", help="对抗样本目录")
    ap.add_argument("--save", type=str, default="adv_grid_labeled.png", help="输出拼图路径")
    ap.add_argument("--nrow", type=int, default=10, help="每行图片数量")
    ap.add_argument("--cell", type=int, default=224, help="单元格尺寸")
    ap.add_argument("--font", type=int, default=14, help="字体大小")
    ap.add_argument("--no-names", action="store_true", help="不显示类别名，只显示ID")
    args = ap.parse_args()

    make_grid_with_labels(
        folder=args.folder,
        save_path=args.save,
        nrow=args.nrow,
        cell_size=args.cell,
        font_size=args.font,
        use_names=not args.no_names,
    )

if __name__ == "__main__":
    main()
