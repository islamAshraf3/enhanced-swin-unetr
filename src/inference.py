import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import LGGDataset, get_transforms, unnormalise
from evaluate import postprocess_mask


def load_model(checkpoint_path, img_size=256, device=None):
    from model import EnhancedSwinUNETR
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedSwinUNETR(img_size=img_size).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]} | Val Dice: {ckpt["val_dice"]:.4f}')
    return model, device


@torch.no_grad()
def predict_tta(model, image_tensor):
    """
    8-fold TTA: original + 3 flips + 4 rotations.
    image_tensor: (1, 3, H, W) on device.
    Returns: (1, 1, H, W) averaged probability map.
    """
    model.eval()
    augmentations = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.flip(x, dims=[2, 3]),
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),
        lambda x: torch.rot90(x, k=2, dims=[2, 3]),
        lambda x: torch.rot90(x, k=3, dims=[2, 3]),
        lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3]),
    ]
    deaugmentations = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.flip(x, dims=[2, 3]),
        lambda x: torch.rot90(x, k=3, dims=[2, 3]),
        lambda x: torch.rot90(x, k=2, dims=[2, 3]),
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),
        lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[3]),
    ]
    probs = []
    for aug, deaug in zip(augmentations, deaugmentations):
        logits = model(aug(image_tensor))
        probs.append(deaug(torch.sigmoid(logits)))
    return torch.stack(probs, dim=0).mean(dim=0)


def predict_single(model, image_path, img_size=256, device=None, use_tta=False):
    """
    Run inference on a single image file.
    Returns: (pred_mask_np H×W float, burden_dict, img_disp H×W×3 float).
    """
    if device is None:
        device = next(model.parameters()).device

    transform = get_transforms(img_size, is_train=False)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    aug = transform(image=img, mask=np.zeros(img.shape[:2], dtype=np.float32))
    img_tensor = aug['image'].unsqueeze(0).to(device)

    if use_tta:
        prob = predict_tta(model, img_tensor)
    else:
        with torch.no_grad():
            prob = torch.sigmoid(model(img_tensor))

    pred_np = prob[0, 0].cpu().numpy()
    pp      = postprocess_mask(pred_np)
    burden  = estimate_tumor_burden(pp)
    img_disp = unnormalise(aug['image'])
    return pred_np, burden, img_disp


def estimate_tumor_burden(pred_mask, pixel_spacing_mm=1.0):
    tumor_px  = int((pred_mask > 0).sum())
    tumor_mm2 = tumor_px * (pixel_spacing_mm ** 2)
    total_px  = pred_mask.shape[0] * pred_mask.shape[1]
    ratio     = tumor_px / total_px
    if ratio == 0:       severity = 0
    elif ratio < 0.01:   severity = 1
    elif ratio < 0.05:   severity = 2
    else:                severity = 3
    return {
        'tumor_area_px'    : tumor_px,
        'tumor_area_mm2'   : round(tumor_mm2, 2),
        'tumor_slice_ratio': round(ratio, 4),
        'severity_score'   : severity,
        'severity_label'   : ['None', 'Mild', 'Moderate', 'Severe'][severity],
    }


@torch.no_grad()
def burden_demo(model, val_df, transform, device, output_dir, n=4, seed=40):
    sample = val_df[val_df['has_tumor'] == 1].sample(n, random_state=seed)
    ds     = LGGDataset(sample, transform=transform)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    imgs, masks, _ = next(iter(loader))
    preds = torch.sigmoid(model(imgs.to(device))).cpu().numpy()

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    headers = ['MRI Slice', 'Ground Truth', 'Raw Prediction', 'Post-proc + Burden']
    for ax, h in zip(axes[0], headers):
        ax.set_title(h, fontsize=10, fontweight='bold', pad=6)

    for row in range(n):
        img_disp = unnormalise(imgs[row])
        mask_np  = masks[row, 0].numpy()
        pred_np  = preds[row, 0]
        pp       = postprocess_mask(pred_np)
        burden   = estimate_tumor_burden(pp)
        axes[row, 0].imshow(img_disp);                              axes[row, 0].axis('off')
        axes[row, 1].imshow(mask_np, cmap='hot', vmin=0, vmax=1);  axes[row, 1].axis('off')
        axes[row, 2].imshow(pred_np, cmap='hot', vmin=0, vmax=1);  axes[row, 2].axis('off')
        axes[row, 3].imshow(pp, cmap='hot')
        axes[row, 3].set_xlabel(
            f'Area:{burden["tumor_area_mm2"]}mm²  '
            f'Ratio:{burden["tumor_slice_ratio"]:.3f}  '
            f'Sev:{burden["severity_label"]}',
            fontsize=8)
        axes[row, 3].axis('off')

    plt.suptitle('Tumor Burden Estimation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tumor_burden.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image',      required=True, help='Path to a single MRI .tif file')
    parser.add_argument('--img_size',   type=int, default=256)
    parser.add_argument('--tta',        action='store_true')
    args = parser.parse_args()

    model, device = load_model(args.checkpoint, img_size=args.img_size)
    pred_np, burden, img_disp = predict_single(
        model, args.image, img_size=args.img_size, device=device, use_tta=args.tta)

    print('\nTumor Burden:')
    for k, v in burden.items():
        print(f'  {k}: {v}')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_disp);                            axes[0].set_title('MRI Slice');     axes[0].axis('off')
    axes[1].imshow(pred_np, cmap='hot', vmin=0, vmax=1); axes[1].set_title('Prediction');    axes[1].axis('off')
    pp = postprocess_mask(pred_np)
    overlay = (img_disp * 255).astype(np.uint8).copy()
    overlay[pp == 1] = (overlay[pp == 1] * 0.5 + np.array([255, 50, 50]) * 0.5).astype(np.uint8)
    axes[2].imshow(overlay);                             axes[2].set_title('Overlay');       axes[2].axis('off')
    axes[2].set_xlabel(
        f'Area: {burden["tumor_area_mm2"]}mm²  Severity: {burden["severity_label"]}', fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
