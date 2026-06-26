import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import timm


# Swin-Tiny encoder channel widths: stage0→stage3
SWIN_CHANNELS = (96, 192, 384, 768)


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.block(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """5 parallel branches: 1x1 | dilation 6 | dilation 12 | dilation 18 | global pool."""

    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))
        for d in dilations[1:]:
            self.branches.append(ASPPConv(in_channels, out_channels, d))
        self.branches.append(ASPPPooling(in_channels, out_channels))
        total_branches = len(dilations) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * total_branches, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class AttentionGate(nn.Module):
    """Soft attention gate: g = gating signal (decoder), x = skip connection (encoder)."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        attention = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * attention


class ResidualDecoderBlock(nn.Module):
    """Upsample -> concat skip -> two conv layers + residual shortcut."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        merged = in_channels + skip_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(merged, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.shortcut = (
            nn.Conv2d(merged, out_channels, kernel_size=1, bias=False)
            if merged != out_channels else nn.Identity()
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        residual = self.shortcut(x)
        x = self.conv2(self.conv1(x))
        return F.relu(x + residual, inplace=True)


class EnhancedSwinUNETR(nn.Module):
    """
    2D Enhanced Swin-UNETR for LGG brain tumor segmentation.

    Encoder: Swin-Tiny (ImageNet pretrained via timm), 4 stages.
    Additions over a plain decoder:
      - ASPP bottleneck with dilations (1, 6, 12, 18)
      - Attention Gates on all three skip connections
      - Residual conv blocks in decoder
      - 4x bilinear head to restore full resolution (patch_size=4 means
        the first encoder stage outputs at H/4 x W/4)
    """

    def __init__(self, img_size=256, aspp_dilations=(1, 6, 12, 18), pretrained=True):
        super().__init__()
        ch = SWIN_CHANNELS  # (96, 192, 384, 768)

        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3],
            img_size=img_size,
        )

        self.aspp = ASPP(ch[3], ch[3], aspp_dilations)

        self.ag2 = AttentionGate(ch[3], ch[2], ch[2] // 2)
        self.ag1 = AttentionGate(ch[2], ch[1], ch[1] // 2)
        self.ag0 = AttentionGate(ch[1], ch[0], ch[0] // 2)

        self.dec1 = ResidualDecoderBlock(ch[3], ch[2], ch[2])
        self.dec2 = ResidualDecoderBlock(ch[2], ch[1], ch[1])
        self.dec3 = ResidualDecoderBlock(ch[1], ch[0], ch[0])

        # Patch size 4 means enc0 is at H/4; upsample 4x to restore full resolution
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(ch[0], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        enc0, enc1, enc2, enc3 = self.backbone(x)
        bottleneck = self.aspp(enc3)
        skip2 = self.ag2(g=bottleneck, x=enc2)
        d1    = self.dec1(bottleneck, skip2)
        skip1 = self.ag1(g=d1, x=enc1)
        d2    = self.dec2(d1, skip1)
        skip0 = self.ag0(g=d2, x=enc0)
        d3    = self.dec3(d2, skip0)
        return self.head(d3)


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    components = {
        'Backbone (Swin-Tiny encoder)': model.backbone,
        'ASPP bottleneck'             : model.aspp,
        'Attention Gates'             : nn.ModuleList([model.ag0, model.ag1, model.ag2]),
        'Decoder blocks'              : nn.ModuleList([model.dec1, model.dec2, model.dec3]),
        'Segmentation head'           : model.head,
    }
    print(f'{"Component":<34} {"Params":>10}')
    print('-' * 46)
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters())
        print(f'  {name:<32}: {n / 1e6:>7.3f} M')
    print('-' * 46)
    print(f'  {"Total":<32}: {total / 1e6:>7.3f} M')
    print(f'  {"Trainable":<32}: {trainable / 1e6:>7.3f} M')


def draw_architecture(output_dir, img_size=256):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis('off')
    ax.set_facecolor('#f8f9fa'); fig.patch.set_facecolor('#f8f9fa')

    def box(x, y, w, h, label, sublabel='', color='#4A90D9', fontsize=9):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.92, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color='white', zorder=4)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.18, sublabel,
                    ha='center', va='center', fontsize=7, color='#d0e8ff', zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5), zorder=2)

    def skip_arrow(x1, y1, x2, y2, label='skip + AG'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.5, linestyle='dashed'), zorder=2)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label,
                fontsize=7, color='#e67e22', ha='center', zorder=4)

    BLUE   = '#2c7bb6'
    GREEN  = '#2ca25f'
    ORANGE = '#d95f02'
    PURPLE = '#7b2d8b'
    GREY   = '#636363'

    s = img_size
    box(0.3, 4.5, 1.6, 0.7, 'Input', f'3×{s}×{s}', GREY)
    for lbl, sub, ypos in [
        ('enc0', f'96×{s//4}×{s//4}',    7.0),
        ('enc1', f'192×{s//8}×{s//8}',   5.8),
        ('enc2', f'384×{s//16}×{s//16}', 4.6),
        ('enc3', f'768×{s//32}×{s//32}', 3.4),
    ]:
        box(2.3, ypos, 2.2, 0.7, lbl, sub, BLUE)
    ax.text(3.4, 8.2, 'Swin-Tiny Encoder\n(ImageNet pretrained)', ha='center', fontsize=9, color=BLUE, fontweight='bold')

    box(5.2, 3.4, 2.2, 0.7, 'ASPP Bottleneck', 'dilations 1,6,12,18', PURPLE)

    for lbl, sub, ypos in [
        ('dec1', f'384×{s//16}×{s//16}', 4.6),
        ('dec2', f'192×{s//8}×{s//8}',   5.8),
        ('dec3', f'96×{s//4}×{s//4}',    7.0),
    ]:
        box(8.3, ypos, 2.2, 0.7, lbl, sub, GREEN)
    ax.text(9.4, 8.2, 'Residual Decoder', ha='center', fontsize=9, color=GREEN, fontweight='bold')

    box(11.2, 7.0, 2.0, 0.7, 'Head + 4× up', f'1×{s}×{s}', ORANGE)

    arrow(1.9, 4.85, 2.3, 7.35)
    for y in [7.0, 5.8, 4.6]: arrow(3.4, y, 3.4, y - 0.9)
    arrow(4.5, 3.75, 5.2, 3.75)
    arrow(7.4, 3.75, 8.3, 4.95)
    for y in [4.95, 6.15]: arrow(9.4, y, 9.4, y + 0.9)
    arrow(10.5, 7.35, 11.2, 7.35)

    skip_arrow(3.5, 4.95, 8.3, 4.95, 'AG2')
    skip_arrow(3.5, 6.15, 8.3, 6.15, 'AG1')
    skip_arrow(3.5, 7.35, 8.3, 7.35, 'AG0')

    legend_items = [
        mpatches.Patch(color=BLUE,      label='Swin-Tiny Encoder (ImageNet)'),
        mpatches.Patch(color=PURPLE,    label='ASPP Bottleneck'),
        mpatches.Patch(color=GREEN,     label='Residual Decoder'),
        mpatches.Patch(color=ORANGE,    label='Head (4× upsample)'),
        mpatches.Patch(color='#e67e22', label='Skip + Attention Gate'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=8, framealpha=0.85, edgecolor='#ccc')
    ax.set_title('Enhanced Swin-UNETR Architecture', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pipeline.png', dpi=180, bbox_inches='tight')
    plt.show()
