"""
Ablation study: Plain U-Net and Attention U-Net baselines vs Enhanced Swin-UNETR.

Results (patient-level 80/20 split, 778 val slices, 256×256, 100 epochs):
  A: Plain U-Net         — Dice: 0.8799  IoU: 0.8513  HD95:  9.22 mm  Acc: 0.9961
  B: Attention U-Net     — Dice: 0.8649  IoU: 0.8358  HD95: 10.65 mm  Acc: 0.9955
  C: Enhanced Swin-UNETR — Dice: 0.9162  IoU: 0.8879  HD95:  5.33 mm  Acc: 0.9977

Key finding: Model B (Attention U-Net) performed *worse* than Model A (Plain U-Net).
Attention gates require semantically rich encoder features to know where to attend.
With a CNN encoder those features lack the global context the gates need, so they
introduce noise rather than signal. The transformer encoder in Model C provides the
global context that makes attention gates effective.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import LGGDataset, unnormalise
from evaluate import dice_score, iou_score, pixel_accuracy, hausdorff95


class UNetConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PlainUNet(nn.Module):
    def __init__(self, in_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        in_ch = in_channels
        for f in features:
            self.encoders.append(UNetConvBlock(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f
        self.bottleneck = UNetConvBlock(features[-1], features[-1] * 2)
        self.upconvs    = nn.ModuleList()
        self.decoders   = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoders.append(UNetConvBlock(f * 2, f))
        self.head = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)


class AttentionGateUNet(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders  = nn.ModuleList()
        self.pools     = nn.ModuleList()
        in_ch = in_channels
        for f in features:
            self.encoders.append(UNetConvBlock(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f
        self.bottleneck = UNetConvBlock(features[-1], features[-1] * 2)
        self.upconvs    = nn.ModuleList()
        self.att_gates  = nn.ModuleList()
        self.decoders   = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.att_gates.append(AttentionGateUNet(F_g=f, F_l=f, F_int=f // 2))
            self.decoders.append(UNetConvBlock(f * 2, f))
        self.head = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, ag, dec, skip in zip(self.upconvs, self.att_gates, self.decoders, reversed(skips)):
            x    = up(x)
            skip = ag(g=x, x=skip)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)


def run_training(model, model_name, ckpt_path, train_loader, val_loader,
                 device, num_epochs=150, lr=1e-4, weight_decay=1e-3, min_lr=1e-6,
                 grad_clip=1.0, dice_weight=0.5, bce_weight=0.5):
    from train import CombinedLoss
    criterion = CombinedLoss(dice_weight, bce_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    scaler    = GradScaler()
    best_dice = 0.0

    print(f'\nTraining {model_name} for {num_epochs} epochs')
    print(f'Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
    print(f'Checkpoint: {ckpt_path}\n')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_dice = tr_iou = 0.0
        for imgs, masks, _ in train_loader:
            imgs  = imgs.to(device,  non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                preds = model(imgs); loss = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                tr_loss += loss.item()
                tr_dice += dice_score(preds, masks)
                tr_iou  += iou_score(preds, masks)
        n_tr = len(train_loader)

        model.eval()
        val_loss = val_dice = val_iou = val_acc = 0.0
        preds_all = []; masks_all = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs  = imgs.to(device); masks = masks.to(device)
                preds = model(imgs); loss = criterion(preds, masks)
                val_loss += loss.item()
                val_dice += dice_score(preds, masks)
                val_iou  += iou_score(preds, masks)
                val_acc  += pixel_accuracy(preds, masks)
                preds_all.append(torch.sigmoid(preds).cpu().numpy())
                masks_all.append(masks.cpu().numpy())
        n_val = len(val_loader)
        import numpy as np
        preds_all = np.concatenate(preds_all, axis=0)
        masks_all = np.concatenate(masks_all, axis=0)
        val_hd    = hausdorff95(preds_all, masks_all)

        scheduler.step()
        improved = val_dice / n_val > best_dice
        if improved:
            best_dice = val_dice / n_val
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_dice': val_dice / n_val, 'val_iou': val_iou / n_val,
                        'val_acc': val_acc / n_val, 'val_hd': val_hd}, ckpt_path)
        tag = ' <- best' if improved else ''
        print(f'Ep {epoch:03d}/{num_epochs}  '
              f'loss {tr_loss/n_tr:.4f}/{val_loss/n_val:.4f}  '
              f'dice {tr_dice/n_tr:.4f}/{val_dice/n_val:.4f}  '
              f'iou {val_iou/n_val:.4f}  hd {val_hd:.2f}  '
              f'[{int(time.time()-t0)}s]{tag}')

    print(f'\nBest {model_name} Val Dice: {best_dice:.4f}')
    return best_dice


@torch.no_grad()
def evaluate_model(model, checkpoint_path, val_loader, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    total_dice = total_iou = total_acc = 0.0
    preds_all = []; masks_all = []
    for imgs, masks, _ in val_loader:
        imgs  = imgs.to(device); masks = masks.to(device)
        preds = model(imgs)
        total_dice += dice_score(preds, masks)
        total_iou  += iou_score(preds, masks)
        total_acc  += pixel_accuracy(preds, masks)
        preds_all.append(torch.sigmoid(preds).cpu().numpy())
        masks_all.append(masks.cpu().numpy())
    import numpy as np
    n  = len(val_loader)
    hd = hausdorff95(np.concatenate(preds_all), np.concatenate(masks_all))
    return total_dice / n, total_iou / n, hd, total_acc / n


def print_ablation_table(results):
    """results: list of (name, dice, iou, hd, acc)"""
    print(f'\n{"Model":<38} {"Dice":>8} {"IoU":>8} {"HD95":>8} {"Acc":>8}')
    print('-' * 74)
    for name, d, i, h, a in results:
        print(f'  {name:<36} {d:>8.4f} {i:>8.4f} {h:>7.2f}mm {a:>8.4f}')
    if len(results) >= 2:
        dA, iA, hA = results[0][1], results[0][2], results[0][3]
        dB, iB, hB = results[1][1], results[1][2], results[1][3]
        print(f'\nGain A→B   : Dice {dB-dA:+.4f} | IoU {iB-iA:+.4f} | HD95 {hB-hA:+.2f}mm')
    if len(results) >= 3:
        dC, iC, hC = results[2][1], results[2][2], results[2][3]
        print(f'Gain B→C   : Dice {dC-dB:+.4f} | IoU {iC-iB:+.4f} | HD95 {hC-hB:+.2f}mm')
        print(f'Gain A→C   : Dice {dC-dA:+.4f} | IoU {iC-iA:+.4f} | HD95 {hC-hA:+.2f}mm')


def visualise_ablation(models_dict, val_df, transform, device, output_dir, n=4, seed=40):
    """
    models_dict: {'Model Name': model, ...}
    Shows predictions side-by-side for each model.
    """
    sample = val_df[val_df['has_tumor'] == 1].sample(n, random_state=seed)
    ds     = LGGDataset(sample, transform=transform)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    imgs, masks, _ = next(iter(loader))

    model_names = list(models_dict.keys())
    n_models    = len(model_names)
    fig, axes   = plt.subplots(n, n_models + 2, figsize=(3.5 * (n_models + 2), 3.5 * n))

    headers = ['MRI', 'Ground Truth'] + model_names
    for ax, h in zip(axes[0], headers):
        ax.set_title(h, fontsize=9, fontweight='bold', pad=6)

    with torch.no_grad():
        all_preds = []
        for model in models_dict.values():
            model.eval()
            preds = torch.sigmoid(model(imgs.to(device))).cpu()
            all_preds.append(preds)

    for row in range(n):
        axes[row, 0].imshow(unnormalise(imgs[row]));                            axes[row, 0].axis('off')
        axes[row, 1].imshow(masks[row, 0].numpy(), cmap='hot', vmin=0, vmax=1); axes[row, 1].axis('off')
        for col, preds in enumerate(all_preds):
            axes[row, col + 2].imshow(preds[row, 0].numpy(), cmap='hot', vmin=0, vmax=1)
            axes[row, col + 2].axis('off')

    plt.suptitle('Ablation Study — Prediction Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved to {output_dir}/ablation_comparison.png')


def main():
    import argparse
    from train import CFG, seed_everything, CombinedLoss
    from model import EnhancedSwinUNETR
    from dataset import build_dataframe, patient_split, get_transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--output_dir',  default='./outputs')
    parser.add_argument('--ckpt_swin',   default=None, help='Pre-trained Enhanced Swin-UNETR checkpoint')
    parser.add_argument('--num_epochs',  type=int, default=150)
    parser.add_argument('--img_size',    type=int, default=256)
    args = parser.parse_args()

    seed_everything()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = build_dataframe(args.data_dir)
    train_df, val_df = patient_split(df)
    train_tf  = get_transforms(args.img_size, is_train=True)
    val_tf    = get_transforms(args.img_size, is_train=False)
    train_loader = DataLoader(LGGDataset(train_df, train_tf),
                              batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(LGGDataset(val_df, val_tf),
                              batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    ckpt_A = os.path.join(args.output_dir, 'ablation_unet.pth')
    ckpt_B = os.path.join(args.output_dir, 'ablation_attunet.pth')

    model_A = PlainUNet().to(device)
    run_training(model_A, 'Plain U-Net', ckpt_A, train_loader, val_loader,
                 device, num_epochs=args.num_epochs)

    model_B = AttentionUNet().to(device)
    run_training(model_B, 'Attention U-Net', ckpt_B, train_loader, val_loader,
                 device, num_epochs=args.num_epochs)

    model_A = PlainUNet().to(device)
    model_B = AttentionUNet().to(device)
    dA, iA, hA, aA = evaluate_model(model_A, ckpt_A, val_loader, device)
    dB, iB, hB, aB = evaluate_model(model_B, ckpt_B, val_loader, device)

    results = [
        ('Plain U-Net',         dA, iA, hA, aA),
        ('Attention U-Net',     dB, iB, hB, aB),
    ]

    if args.ckpt_swin:
        model_C = EnhancedSwinUNETR(img_size=args.img_size).to(device)
        dC, iC, hC, aC = evaluate_model(model_C, args.ckpt_swin, val_loader, device)
        results.append(('Enhanced Swin-UNETR', dC, iC, hC, aC))

    print_ablation_table(results)

    models_dict = {'Plain U-Net': model_A, 'Attention U-Net': model_B}
    if args.ckpt_swin:
        models_dict['Enhanced Swin-UNETR'] = model_C
    visualise_ablation(models_dict, val_df, val_tf, device, args.output_dir)


if __name__ == '__main__':
    main()
