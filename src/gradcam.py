import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import LGGDataset, unnormalise


class GradCAM:
    """
    Grad-CAM hooked at dec3.conv2 — the last decoder block before the 4× head upsample,
    giving the highest-resolution feature map (H/4 × W/4) with the most spatial detail.
    """

    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.model.dec3.conv2.register_forward_hook(forward_hook)
        h2 = self.model.dec3.conv2.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, img_tensor):
        """
        img_tensor: (1, 3, H, W) on device.
        Returns: heatmap (H, W) numpy array in [0, 1].
        """
        img_size = img_tensor.shape[-2:]
        self._register_hooks()
        self.model.zero_grad()
        self.model.eval()
        output = self.model(img_tensor)
        loss   = torch.sigmoid(output).mean()
        loss.backward()
        self._remove_hooks()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=img_size, mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualise_gradcam(model, val_df, transform, device, output_dir, n=4, seed=40):
    gradcam = GradCAM(model)
    sample  = val_df[val_df['has_tumor'] == 1].sample(n, random_state=seed)
    ds      = LGGDataset(sample, transform=transform)
    loader  = DataLoader(ds, batch_size=1, shuffle=False)

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    headers = ['MRI Slice', 'Ground Truth', 'Prediction', 'Grad-CAM Heatmap']
    for ax, h in zip(axes[0], headers):
        ax.set_title(h, fontsize=11, fontweight='bold', pad=6)

    for row, (imgs, masks, _) in enumerate(loader):
        if row >= n:
            break
        img_tensor = imgs.to(device)
        cam        = gradcam.generate(img_tensor)

        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor)).cpu()

        img_disp = unnormalise(imgs[0])
        mask_np  = masks[0, 0].numpy()
        pred_np  = pred[0, 0].numpy()

        heatmap_colored = plt.cm.jet(cam)[:, :, :3]
        overlay = img_disp * 0.5 + heatmap_colored * 0.5

        axes[row, 0].imshow(img_disp);                              axes[row, 0].axis('off')
        axes[row, 1].imshow(mask_np, cmap='hot', vmin=0, vmax=1);  axes[row, 1].axis('off')
        axes[row, 2].imshow(pred_np, cmap='hot', vmin=0, vmax=1);  axes[row, 2].axis('off')
        axes[row, 3].imshow(overlay);                               axes[row, 3].axis('off')

    plt.suptitle('Grad-CAM — Regions Influencing Prediction', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gradcam.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved to {output_dir}/gradcam.png')


def main():
    import argparse
    from train import CFG, seed_everything
    from model import EnhancedSwinUNETR
    from dataset import build_dataframe, patient_split, get_transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--img_size',   type=int, default=256)
    parser.add_argument('--n',          type=int, default=4)
    args = parser.parse_args()

    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = build_dataframe(args.data_dir)
    _, val_df = patient_split(df)
    val_transforms = get_transforms(args.img_size, is_train=False)

    model = EnhancedSwinUNETR(img_size=args.img_size).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    visualise_gradcam(model, val_df, val_transforms, device, args.output_dir, n=args.n)


if __name__ == '__main__':
    main()
