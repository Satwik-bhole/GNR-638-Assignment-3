"""  
PSPNet (Pyramid Scene Parsing Network) - built from scratch.

Reference: Zhao et al., "Pyramid Scene Parsing Network", CVPR 2017.

The core idea behind PSPNet is that standard FCNs miss out on global scene
context. A Pyramid Pooling Module captures scene-level priors by aggregating
features at four coarse spatial grids and folding that context back in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ── small helper so we don't keep repeating Conv-BN-ReLU everywhere ──────────
def _conv_bn_relu(in_ch, out_ch, ksize, pad=0, has_bias=False):
    """A tiny helper that stacks a convolution, batch-norm, and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=pad, bias=has_bias),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ── single pooling branch used inside the pyramid module ─────────────────────
class _PoolBranch(nn.Module):
    """One branch of the pyramid: pool to a fixed grid, project, then resize."""

    def __init__(self, in_ch, out_ch, grid_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=grid_size)
        self.proj = _conv_bn_relu(in_ch, out_ch, ksize=1)  # 1x1 projection

    def forward(self, x):
        target_h, target_w = x.shape[2], x.shape[3]
        out = self.pool(x)
        out = self.proj(out)
        # bring it back to the spatial size of the incoming feature map
        out = F.interpolate(out, size=(target_h, target_w),
                            mode="bilinear", align_corners=True)
        return out


class MultiScalePooling(nn.Module):
    """
    Pyramid Pooling Module — the signature component of PSPNet.

    We pool the feature map at four different grid resolutions (1x1, 2x2,
    3x3, 6x6), project each to `reduce_ch` channels via a 1x1 conv, then
    upsample everything back and concatenate with the original features.

    After concatenation the channel count becomes:
        in_ch + len(grid_sizes) * reduce_ch
    """

    def __init__(self, in_ch, reduce_ch, grid_sizes=(1, 2, 3, 6)):
        super().__init__()
        # each grid size gets its own independent branch
        self.pool_branches = nn.ModuleList(
            [_PoolBranch(in_ch, reduce_ch, gs) for gs in grid_sizes]
        )
        # total output channels after concatenation
        self.out_channels = in_ch + reduce_ch * len(grid_sizes)

    def forward(self, x):
        # collect the multi-scale context from every branch
        multi_scale = [branch(x) for branch in self.pool_branches]
        # append the original feature map so we keep fine-grained detail too
        return torch.cat([x] + multi_scale, dim=1)


# ── segmentation head (shared design for main & aux) ─────────────────────────
def _make_seg_head(in_ch, mid_ch, num_classes, drop_rate=0.1):
    """Build a small classification head: 3x3 conv -> BN -> ReLU -> drop -> 1x1."""
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=drop_rate),
        nn.Conv2d(mid_ch, num_classes, kernel_size=1),
    )


# ── deep stem: three 3×3 convs instead of one 7×7 (as in the paper) ──────────
class _DeepStem(nn.Module):
    """
    The PSPNet authors replace ResNet's single 7×7 conv with three stacked
    3×3 convolutions (3→64, 64→64, 64→128) followed by a max-pool.
    This captures the same receptive field with fewer parameters and lets
    early features be richer.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


# ── backbone: dilated ResNet-50 ──────────────────────────────────────────────
def _build_dilated_resnet50():
    """
    Load a pretrained ResNet-50 and swap the last two stages to use
    dilated (atrous) convolutions so that the output stride is 8
    instead of the default 32.  This retains spatial detail that the
    pyramid module relies on.
    """
    backbone = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V1,
        replace_stride_with_dilation=[False, True, True],
    )
    return backbone


class PSPNetScratch(nn.Module):
    """
    My from-scratch PSPNet.

    Pipeline:
        image --> deep stem --> dilated ResNet stages --> pyramid pooling --> classifier
                                       |
                                 (aux head, only while training)

    The stem uses three 3×3 convs (paper's "modified ResNet") rather than
    the standard single 7×7 conv.  The pretrained layer1 expects 64-ch
    input, but our deep stem outputs 128 ch, so we add a small 1×1
    adapter to bridge the gap.
    """

    def __init__(self, num_classes=21, use_aux=False):
        super().__init__()
        self.use_aux = use_aux

        # ── backbone ──
        backbone = _build_dilated_resnet50()

        # paper-style deep stem: three 3×3 convs (outputs 128 channels)
        self.stem = _DeepStem()

        # bridge from 128-ch stem to 64-ch expected by layer1's first block
        self.stem_adapter = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # residual stages from the pretrained backbone
        # (stage 3 & 4 already use dilation thanks to replace_stride_with_dilation)
        self.res_stage1 = backbone.layer1   # 64  -> 256  ch
        self.res_stage2 = backbone.layer2   # 256 -> 512  ch
        self.res_stage3 = backbone.layer3   # 512 -> 1024 ch  (dilated)
        self.res_stage4 = backbone.layer4   # 1024-> 2048 ch  (dilated)

        # ── pyramid pooling ──
        # 2048 input channels, each of the 4 branches reduces to 512
        # output: 2048 + 4*512 = 4096 channels
        self.pyramid_pool = MultiScalePooling(
            in_ch=2048, reduce_ch=512, grid_sizes=(1, 2, 3, 6)
        )

        # ── main segmentation head ──
        self.seg_head = _make_seg_head(
            in_ch=self.pyramid_pool.out_channels,  # 4096
            mid_ch=512, num_classes=num_classes, drop_rate=0.1
        )

        # ── auxiliary head (taps off stage 3 for deeper supervision) ──
        if self.use_aux:
            self.aux_head = _make_seg_head(
                in_ch=1024, mid_ch=256,
                num_classes=num_classes, drop_rate=0.1
            )

        # give our freshly-created layers a proper starting point
        self._kaiming_init_heads()

    # ------------------------------------------------------------------
    def _kaiming_init_heads(self):
        """Kaiming-normal init for every conv in our new heads + stem."""
        modules_to_init = [self.stem, self.stem_adapter,
                           self.pyramid_pool, self.seg_head]
        if self.use_aux:
            modules_to_init.append(self.aux_head)

        for mod_group in modules_to_init:
            for m in mod_group.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                           nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]

        # run the image through the deep stem + adapter + dilated backbone
        feat = self.stem(x)
        feat = self.stem_adapter(feat)
        feat = self.res_stage1(feat)
        feat = self.res_stage2(feat)
        feat_s3 = self.res_stage3(feat)   # kept aside for the aux branch
        feat_s4 = self.res_stage4(feat_s3)

        # aggregate multi-scale context through the pyramid module
        feat_ppm = self.pyramid_pool(feat_s4)

        # predict per-pixel class scores and resize to the original image size
        logits = self.seg_head(feat_ppm)
        logits = F.interpolate(logits, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=True)

        # during training, also produce an auxiliary prediction from stage 3
        if self.training and self.use_aux:
            aux_logits = self.aux_head(feat_s3)
            aux_logits = F.interpolate(aux_logits, size=(orig_h, orig_w),
                                       mode="bilinear", align_corners=True)
            return logits, aux_logits

        return logits


# ── quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    net = PSPNetScratch(num_classes=21, use_aux=True)

    net.train()
    sample = torch.randn(2, 3, 473, 473)
    main, aux = net(sample)
    print(f"[train] main: {main.shape}, aux: {aux.shape}")

    net.eval()
    with torch.no_grad():
        pred = net(sample)
    print(f"[eval]  pred: {pred.shape}")
