import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt, label as cc_label, binary_fill_holes

from dataset import LGGDataset, unnormalise


def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred   = (torch.sigmoid(pred) > threshold).float()
    inter  = (pred * target).sum(dim=(1, 2, 3))
    union  = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred  = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


def pixel_accuracy(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    return (pred == target).float().mean().item()


def hausdorff95(pred_np, target_np):
    pred_bin   = (pred_np > 0.5).astype(np.uint8)
    target_bin = target_np.astype(np.uint8)
    dists = []
    for p, t in zip(pred_bin[:, 0], target_bin[:, 0]):
        if p.max() == 0 and t.max() == 0:
            dists.append(0.0); continue
        if p.max() == 0 or t.max() == 0:
            dists.append(100.0); continue
        dt_p = distance_transform_edt(1 - p)
        dt_t = distance_transform_edt(1 - t)
        dists.append(float(max(np.percentile(dt_t[p > 0], 95),
                               np.percentile(dt_p[t > 0], 95))))
    return float(np.mean(dists))


@torch.no_grad()
def full_evaluation(model, loader, device):
    model.eval()
    records = []
    for imgs, masks, metas in loader:
        imgs  = imgs.to(device)
        preds = model(imgs)
        preds_np = torch.sigmoid(preds).cpu().numpy()
        masks_np = masks.numpy()
        for i in range(len(preds)):
            p = preds[i:i+1]
            m = masks[i:i+1].to(device)
            records.append({
                'patient_id': metas['patient_id'][i],
                'image_path': metas['image_path'][i],
                'has_tumor' : metas['has_tumor'][i].item(),
                'dice'      : dice_score(p, m),
                'iou'       : iou_score(p, m),
                'accuracy'  : pixel_accuracy(p, m),
                'hausdorff' : hausdorff95(preds_np[i:i+1], masks_np[i:i+1]),
            })
    return pd.DataFrame(records)


@torch.no_grad()
def evaluate_tumor_only(model, val_df, transform, device, batch_size=16, num_workers=2):
    tumor_val_df  = val_df[val_df['has_tumor'] == 1].reset_index(drop=True)
    tumor_loader  = DataLoader(LGGDataset(tumor_val_df, transform),
                               batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_dice = total_iou = total_acc = 0.0
    for imgs, masks, _ in tumor_loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        total_dice += dice_score(preds, masks)
        total_iou  += iou_score(preds, masks)
        total_acc  += pixel_accuracy(preds, masks)
    n = len(tumor_loader)
    print(f'Tumor-only evaluation  ({len(tumor_val_df)} slices)')
    print(f'  Dice     : {total_dice / n:.4f}')
    print(f'  IoU      : {total_iou / n:.4f}')
    print(f'  Accuracy : {total_acc / n:.4f}')
    return total_dice / n, total_iou / n, total_acc / n


def postprocess_mask(pred_np, min_pixels=50):
    binary  = (pred_np > 0.5).astype(np.uint8)
    filled  = binary_fill_holes(binary).astype(np.uint8)
    labeled, n_comp = cc_label(filled)
    cleaned = np.zeros_like(filled)
    for c in range(1, n_comp + 1):
        comp = (labeled == c)
        if comp.sum() >= min_pixels:
            cleaned[comp] = 1
    return cleaned


@torch.no_grad()
def evaluate_with_postprocessing(model, loader, device, min_pixels=50):
    model.eval()
    dr = di = pp_d = pp_i = total_hd_raw = total_hd_pp = 0.0
    n = 0
    for imgs, masks, _ in loader:
        imgs  = imgs.to(device)
        preds = torch.sigmoid(model(imgs)).cpu()
        masks_np = masks.numpy()
        preds_np = preds.numpy()
        for i in range(len(preds)):
            p = preds[i:i+1]
            m = masks[i:i+1]
            inter = (p * m).sum(); union = p.sum() + m.sum()
            dr += ((2 * inter + 1e-6) / (union + 1e-6)).item()
            i_iou = (p * m).sum(); u_iou = (p + m - p * m).sum()
            di += ((i_iou + 1e-6) / (u_iou + 1e-6)).item()
            total_hd_raw += hausdorff95(preds_np[i:i+1], masks_np[i:i+1])
            pp   = postprocess_mask(preds_np[i, 0], min_pixels=min_pixels)
            pp_t = torch.tensor(pp).unsqueeze(0).unsqueeze(0).float()
            inter2 = (pp_t * m).sum(); union2 = pp_t.sum() + m.sum()
            pp_d += ((2 * inter2 + 1e-6) / (union2 + 1e-6)).item()
            i2 = (pp_t * m).sum(); u2 = (pp_t + m - pp_t * m).sum()
            pp_i += ((i2 + 1e-6) / (u2 + 1e-6)).item()
            total_hd_pp += hausdorff95(pp_t.numpy(), masks_np[i:i+1])
            n += 1
    print(f'Post-processing comparison ({n} slices):')
    print(f'  {"Metric":<8} {"Raw":>10} {"Post-proc":>12} {"Gain":>8}')
    print(f'  {"-" * 40}')
    print(f'  {"Dice":<8} {dr / n:>10.4f} {pp_d / n:>12.4f} {pp_d / n - dr / n:>+8.4f}')
    print(f'  {"IoU":<8} {di / n:>10.4f} {pp_i / n:>12.4f} {pp_i / n - di / n:>+8.4f}')
    print(f'  {"HD95":<8} {total_hd_raw / n:>10.2f} {total_hd_pp / n:>12.2f} {total_hd_pp / n - total_hd_raw / n:>+8.2f}')


def print_final_summary(df_results, ckpt_epoch):
    tumor_df = df_results[df_results['has_tumor'] == 1]
    print('=' * 60)
    print('FINAL EVALUATION SUMMARY')
    print('=' * 60)
    print(f'  Checkpoint  : epoch {ckpt_epoch}')
    print(f'  Val slices  : {len(df_results)} ({len(tumor_df)} with tumor)')
    print(f'  Patients    : {df_results["patient_id"].nunique()}')
    print()
    print(f'  All-slice Dice     : {df_results["dice"].mean():.4f}')
    print(f'  All-slice IoU      : {df_results["iou"].mean():.4f}')
    print(f'  All-slice HD95     : {df_results["hausdorff"].mean():.2f} mm')
    print(f'  All-slice Accuracy : {df_results["accuracy"].mean():.4f}')
    print()
    print(f'  Tumor-only Dice    : {tumor_df["dice"].mean():.4f}')
    print(f'  Tumor-only IoU     : {tumor_df["iou"].mean():.4f}')
    print(f'  Tumor-only HD95    : {tumor_df["hausdorff"].mean():.2f} mm')
    print('=' * 60)


def visualise_predictions(model, val_df, transform, device, output_dir, n=6, seed=40):
    sample = val_df[val_df['has_tumor'] == 1].sample(n, random_state=seed)
    ds     = LGGDataset(sample, transform=transform)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    imgs, masks, _ = next(iter(loader))

    with torch.no_grad():
        preds = torch.sigmoid(model(imgs.to(device))).cpu()

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    headers = ['MRI Slice', 'Ground Truth', 'Prediction', 'Overlay']
    for ax, h in zip(axes[0], headers):
        ax.set_title(h, fontsize=11, fontweight='bold', pad=6)

    for row in range(n):
        img_disp = unnormalise(imgs[row])
        mask_np  = masks[row, 0].numpy()
        pred_np  = preds[row, 0].numpy()
        pred_bin = (pred_np > 0.5).astype(np.uint8)
        overlay  = (img_disp * 255).astype(np.uint8).copy()
        overlay[pred_bin == 1] = (overlay[pred_bin == 1] * 0.5 + np.array([255, 50, 50]) * 0.5).astype(np.uint8)
        p_t = torch.tensor(pred_np).unsqueeze(0).unsqueeze(0)
        m_t = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
        inter = (p_t * m_t).sum(); union = p_t.sum() + m_t.sum()
        d = ((2 * inter + 1e-6) / (union + 1e-6)).item()
        axes[row, 0].imshow(img_disp);                            axes[row, 0].axis('off')
        axes[row, 1].imshow(mask_np, cmap='hot', vmin=0, vmax=1); axes[row, 1].axis('off')
        axes[row, 2].imshow(pred_np, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].set_xlabel(f'Dice: {d:.3f}', fontsize=8);   axes[row, 2].axis('off')
        axes[row, 3].imshow(overlay);                             axes[row, 3].axis('off')

    plt.suptitle('Qualitative Predictions — Enhanced Swin-UNETR', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


def main(checkpoint_path, data_dir, output_dir, img_size=256, batch_size=16, num_workers=2):
    from train import CFG, seed_everything
    from model import EnhancedSwinUNETR
    from dataset import build_dataframe, patient_split, get_transforms

    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = build_dataframe(data_dir)
    _, val_df = patient_split(df, val_frac=0.20, seed=42)
    val_transforms = get_transforms(img_size, is_train=False)
    val_loader = DataLoader(
        LGGDataset(val_df, val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = EnhancedSwinUNETR(img_size=img_size).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    df_results = full_evaluation(model, val_loader, device)
    print_final_summary(df_results, ckpt['epoch'])

    visualise_predictions(model, val_df, val_transforms, device, output_dir)
    evaluate_with_postprocessing(model, val_loader, device)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--img_size',   type=int, default=256)
    args = parser.parse_args()
    main(args.checkpoint, args.data_dir, args.output_dir, args.img_size)
