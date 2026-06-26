import os
import time
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from dataset import build_dataframe, patient_split, get_transforms, LGGDataset
from model import EnhancedSwinUNETR
from evaluate import dice_score, iou_score, pixel_accuracy, hausdorff95

warnings.filterwarnings('ignore')


class CFG:
    DATA_DIR         = '/content/lgg-mri-segmentation/kaggle_3m'
    OUTPUT_DIR       = '/content/drive/MyDrive/brain-tumor'
    CKPT_PATH        = '/content/drive/MyDrive/brain-tumor/best_model_imagenet.pth'

    IMG_SIZE         = 256
    IN_CHANNELS      = 3
    BATCH_SIZE       = 16
    NUM_WORKERS      = 2
    VAL_FRAC         = 0.20
    SEED             = 42

    MEAN             = (0.485, 0.456, 0.406)
    STD              = (0.229, 0.224, 0.225)

    ASPP_DILATIONS   = (1, 6, 12, 18)

    NUM_EPOCHS       = 100
    LR               = 1e-4
    WEIGHT_DECAY     = 1e-3
    MIN_LR           = 1e-6
    GRAD_CLIP        = 1.0

    DICE_WEIGHT      = 0.5
    BCE_WEIGHT       = 0.5

    PIXEL_SPACING_MM = 1.0
    MIN_TUMOR_PIXELS = 50


def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / \
               (pred.sum() + target.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=CFG.DICE_WEIGHT, bce_weight=CFG.BCE_WEIGHT):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return (self.dice_weight * self.dice(pred, target) +
                self.bce_weight  * self.bce(pred, target))


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = total_dice = total_iou = 0.0
    for imgs, masks, _ in loader:
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda'):
            preds = model(imgs)
            loss  = criterion(preds, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            total_loss += loss.item()
            total_dice += dice_score(preds, masks)
            total_iou  += iou_score(preds,  masks)
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = total_acc = 0.0
    preds_all = []; masks_all = []
    for imgs, masks, _ in loader:
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        total_loss += loss.item()
        total_dice += dice_score(preds, masks)
        total_iou  += iou_score(preds,  masks)
        total_acc  += pixel_accuracy(preds, masks)
        preds_all.append(torch.sigmoid(preds).cpu().numpy())
        masks_all.append(masks.cpu().numpy())
    n = len(loader)
    preds_all = np.concatenate(preds_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)
    hd = hausdorff95(preds_all, masks_all)
    return total_loss / n, total_dice / n, total_iou / n, total_acc / n, hd


def plot_training_curves(history, save_path):
    epochs = range(1, len(history['tr_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history['tr_loss'],  label='Train', color='#2c7bb6')
    axes[0].plot(epochs, history['val_loss'], label='Val',   color='#d7191c')
    axes[0].set_title('Combined Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['tr_dice'],  label='Train', color='#2c7bb6')
    axes[1].plot(epochs, history['val_dice'], label='Val',   color='#d7191c')
    best_epoch = int(np.argmax(history['val_dice'])) + 1
    best_val   = max(history['val_dice'])
    axes[1].axvline(best_epoch, color='green', linestyle='--', alpha=0.7,
                    label=f'Best ep {best_epoch}: {best_val:.4f}')
    axes[1].set_title('Dice Score', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history['val_iou'], label='Val IoU',      color='#1a9641')
    axes[2].plot(epochs, history['val_acc'], label='Val Accuracy',  color='#756bb1', linestyle='--')
    axes[2].set_title('Val IoU & Accuracy', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle('Training History — Enhanced Swin-UNETR', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    seed_everything()
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    df = build_dataframe(CFG.DATA_DIR)
    train_df, val_df = patient_split(df, val_frac=CFG.VAL_FRAC, seed=CFG.SEED)
    print(f'Train: {len(train_df)} slices | Val: {len(val_df)} slices')

    train_transforms = get_transforms(CFG.IMG_SIZE, is_train=True)
    val_transforms   = get_transforms(CFG.IMG_SIZE, is_train=False)

    train_loader = DataLoader(
        LGGDataset(train_df, train_transforms),
        batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        LGGDataset(val_df, val_transforms),
        batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model     = EnhancedSwinUNETR(img_size=CFG.IMG_SIZE, aspp_dilations=CFG.ASPP_DILATIONS).to(device)
    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.NUM_EPOCHS, eta_min=CFG.MIN_LR)
    scaler    = GradScaler()
    history   = defaultdict(list)
    best_dice = 0.0

    print(f'Training for {CFG.NUM_EPOCHS} epochs')
    print(f'Optimizer : AdamW  (lr={CFG.LR}, wd={CFG.WEIGHT_DECAY})')
    print(f'Scheduler : CosineAnnealingLR  (eta_min={CFG.MIN_LR})')
    print(f'Loss      : {CFG.DICE_WEIGHT}xDice + {CFG.BCE_WEIGHT}xBCE')
    print()

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_dice, tr_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device)

        val_loss, val_dice, val_iou, val_acc, val_hd = validate(
            model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        for k, v in zip(
            ['tr_loss', 'val_loss', 'tr_dice', 'val_dice', 'tr_iou', 'val_iou', 'val_acc'],
            [tr_loss,   val_loss,   tr_dice,   val_dice,   tr_iou,   val_iou,   val_acc]):
            history[k].append(v)

        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_dice'   : val_dice,
                'val_iou'    : val_iou,
                'val_acc'    : val_acc,
                'val_hd'     : val_hd,
                'history'    : dict(history),
            }, CFG.CKPT_PATH)

        tag = ' <- best' if improved else ''
        print(
            f'Ep {epoch:03d}/{CFG.NUM_EPOCHS}  '
            f'loss {tr_loss:.4f}/{val_loss:.4f}  '
            f'dice {tr_dice:.4f}/{val_dice:.4f}  '
            f'iou {val_iou:.4f}  '
            f'hd {val_hd:.2f}  '
            f'acc {val_acc:.4f}  '
            f'lr {lr_now:.2e}  '
            f'[{elapsed:.0f}s]{tag}'
        )

    print(f'\nBest Val Dice : {best_dice:.4f}')
    print(f'Checkpoint    : {CFG.CKPT_PATH}')

    plot_training_curves(history, f'{CFG.OUTPUT_DIR}/training_curves.png')


if __name__ == '__main__':
    main()
