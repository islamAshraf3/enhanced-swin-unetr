import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def build_dataframe(data_dir):
    rows = []
    for patient_dir in sorted(Path(data_dir).iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        mask_lookup = {
            m.stem.replace('_mask', ''): m
            for m in patient_dir.glob('*_mask.tif')
        }
        for img_path in sorted(patient_dir.glob('*.tif')):
            if '_mask' in img_path.stem:
                continue
            stem = img_path.stem
            if stem in mask_lookup:
                rows.append({
                    'patient_id': pid,
                    'image_path': str(img_path),
                    'mask_path' : str(mask_lookup[stem]),
                })
    df = pd.DataFrame(rows)
    has_tumor = []
    for mp in df['mask_path']:
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        has_tumor.append(1 if (m is not None and m.max() > 0) else 0)
    df['has_tumor'] = has_tumor
    return df


def patient_split(df, val_frac=0.20, seed=42):
    patients = df['patient_id'].unique()
    train_patients, val_patients = train_test_split(
        patients, test_size=val_frac, random_state=seed, shuffle=True)
    assert len(set(train_patients) & set(val_patients)) == 0, \
        'DATA LEAKAGE: patient appears in both splits!'
    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df['patient_id'].isin(val_patients)  ].reset_index(drop=True)
    return train_df, val_df


def get_transforms(img_size=256, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.15),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


class LGGDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        image  = cv2.imread(record['image_path'])
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {record["image_path"]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(record['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Cannot read mask: {record["mask_path"]}')
        mask = (mask > 127).astype(np.float32)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']
        mask = mask.unsqueeze(0).float()
        meta = {
            'patient_id': record['patient_id'],
            'image_path': record['image_path'],
            'has_tumor' : int(record['has_tumor']),
        }
        return image, mask, meta


def unnormalise(tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD ).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def visualise_raw_samples(df, output_dir, n=5, seed=42):
    sample = df[df['has_tumor'] == 1].sample(n, random_state=seed)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n))
    col_titles = ['T1-weighted MRI', 'Ground Truth Mask', 'Overlay (contour)']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    for row_idx, (_, record) in enumerate(sample.iterrows()):
        img  = cv2.cvtColor(cv2.imread(record['image_path']), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(record['mask_path'], cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 127).astype(np.uint8)
        overlay = img.copy()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 50, 50), 2)
        axes[row_idx, 0].imshow(img);                                          axes[row_idx, 0].axis('off')
        axes[row_idx, 0].set_ylabel(record['patient_id'], rotation=0, labelpad=60, va='center', fontsize=8)
        axes[row_idx, 1].imshow(binary_mask, cmap='hot', vmin=0, vmax=1);     axes[row_idx, 1].axis('off')
        axes[row_idx, 2].imshow(overlay);                                      axes[row_idx, 2].axis('off')
        tumor_px = binary_mask.sum()
        axes[row_idx, 2].set_xlabel(
            f'Tumor area: {tumor_px} px  ({tumor_px / (img.shape[0] * img.shape[1]) * 100:.2f}% of slice)',
            fontsize=8)
    plt.suptitle('Raw Dataset Samples LGG MRI', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/raw_samples.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualise_augmentation(df, output_dir, train_transforms, val_transforms, n_samples=4, seed=42):
    sample_rows = df[df['has_tumor'] == 1].sample(n_samples, random_state=seed)
    fig, axes = plt.subplots(n_samples, 5, figsize=(17, 3.5 * n_samples))
    headers = ['Original', 'Augmented run 1', 'Augmented run 2', 'Mask (run 1)', 'Mask (run 2)']
    for ax, h in zip(axes[0], headers):
        ax.set_title(h, fontsize=10, fontweight='bold', pad=6)
    for row, (_, record) in enumerate(sample_rows.iterrows()):
        img_raw  = cv2.cvtColor(cv2.imread(record['image_path']), cv2.COLOR_BGR2RGB)
        mask_raw = (cv2.imread(record['mask_path'], cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        aug0 = val_transforms(image=img_raw, mask=mask_raw)
        aug1 = train_transforms(image=img_raw, mask=mask_raw)
        aug2 = train_transforms(image=img_raw, mask=mask_raw)
        axes[row, 0].imshow(unnormalise(aug0['image'])); axes[row, 0].axis('off')
        axes[row, 1].imshow(unnormalise(aug1['image'])); axes[row, 1].axis('off')
        axes[row, 2].imshow(unnormalise(aug2['image'])); axes[row, 2].axis('off')
        axes[row, 3].imshow(aug1['mask'].numpy(), cmap='hot', vmin=0, vmax=1); axes[row, 3].axis('off')
        axes[row, 4].imshow(aug2['mask'].numpy(), cmap='hot', vmin=0, vmax=1); axes[row, 4].axis('off')
    plt.suptitle('Augmentation Stochasticity Check', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/augmentation_check.png', dpi=150, bbox_inches='tight')
    plt.show()
