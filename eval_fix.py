#!/usr/bin/env python3
"""
CondDiff-AMO — standalone evaluator.

Usage:
    python eval_all_in_one.py \
        --checkpoint /path/to/best_miou_ckpt.pth \
        --coco-img-dir /path/to/images \
        --cocoa-val-json /path/to/val.json

    # Chạy toàn bộ validation (không giới hạn sample)
    python eval_all_in_one.py --max-samples 0

    # Smoke test nhanh
    python eval_all_in_one.py --max-samples 64 --steps 5
"""

import argparse
import gc
import json
import math
import os
import sys
import traceback

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ============================================================
# CONFIG — chỉnh default path ở đây, hoặc override qua CLI
# ============================================================
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent

DEFAULT_DATA_ROOT       = _PROJECT_ROOT / "data"
DEFAULT_CHECKPOINT      = _PROJECT_ROOT / "checkpoints" / "best_miou_ckpt.pth"
DEFAULT_COCO_IMG_DIR    = DEFAULT_DATA_ROOT / "coco_amodal" / "cocoa_images_extracted"
DEFAULT_COCOA_VAL_JSON  = DEFAULT_DATA_ROOT / "coco_amodal" / "COCO_amodal_val2014.json"
DEFAULT_OUTPUT_DIR      = _PROJECT_ROOT / "eval_results"


# ============================================================
# Runtime constants
# ============================================================
DEFAULT_STEPS       = 10
DEFAULT_MAX_SAMPLES = 0          # <= 0 = không giới hạn
DEFAULT_BATCH_SIZE  = 8
DEFAULT_WORKERS     = 2
IMAGE_SIZE          = (256, 256)
TIMESTEPS           = 1000


# ============================================================
# Dataset (COCOA only)
# ============================================================
class AmodalDataset(Dataset):
    def __init__(self, data_dir, mode="cocoa_val", image_size=(256, 256),
                 coco_img_dir=None):
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.coco_img_dir = coco_img_dir
        self.samples = []

        if not self.mode.startswith("cocoa"):
            raise ValueError(f"Chỉ hỗ trợ cocoa*, nhận: {self.mode}")
        if not self.coco_img_dir:
            raise ValueError("coco_img_dir là bắt buộc.")

        print(f"⏳ Đang phân tích JSON: {self.mode.upper()}")
        self.available_images = set(os.listdir(self.coco_img_dir))
        with open(self.data_dir, 'r') as f:
            cocoa_data = json.load(f)

        skipped_img, skipped_degen = 0, 0

        for ann in cocoa_data['annotations']:
            img_filename = ann['url'].split('/')[-1]
            if img_filename not in self.available_images:
                skipped_img += 1
                continue

            depth_constraints = ann.get('depth_constraint', '')
            occluders_of = {}
            if depth_constraints:
                for pair in depth_constraints.split(','):
                    if '-' in pair:
                        front, back = pair.split('-')
                        occluders_of.setdefault(int(back), []).append(int(front))

            for region in ann['regions']:
                if region.get('isStuff', 0) == 1:
                    continue
                region_order = region.get('order')
                if region_order is None:
                    continue
                seg = region.get('segmentation')
                if not seg:
                    skipped_degen += 1
                    continue
                self.samples.append({
                    'img_filename': img_filename,
                    'region': region,
                    'all_regions': ann['regions'],
                    'occluders': occluders_of.get(region_order, []),
                })

        self.num_samples = len(self.samples)
        print(f"✅ Loaded {self.num_samples} objects.")
        if skipped_img:
            print(f"⚠️ Bỏ qua {skipped_img} ảnh thiếu.")
        if skipped_degen:
            print(f"⚠️ Bỏ qua {skipped_degen} region degenerate.")

    def _poly_to_mask(self, polygons, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        if isinstance(polygons[0], list):
            pts = [np.array(p, np.int32).reshape((-1, 2)) for p in polygons]
        else:
            pts = [np.array(polygons, np.int32).reshape((-1, 2))]
        cv2.fillPoly(mask, pts, 1)
        return mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.coco_img_dir, s['img_filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        amodal_mask = self._poly_to_mask(s['region']['segmentation'], h_orig, w_orig)
        occluder_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for occ_order in s['occluders']:
            for r in s['all_regions']:
                if r.get('order') == occ_order:
                    occ = self._poly_to_mask(r['segmentation'], h_orig, w_orig)
                    occluder_mask = np.logical_or(occluder_mask, occ).astype(np.uint8)
                    break

        modal_mask = amodal_mask.copy()
        modal_mask[occluder_mask == 1] = 0

        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        m_v = cv2.resize(modal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        m_a = cv2.resize(amodal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))
        m_v = np.expand_dims(m_v.astype(np.float32), axis=0)
        m_a = np.expand_dims(m_a.astype(np.float32), axis=0)
        return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a), s['img_filename']


# ============================================================
# Geometry
# ============================================================
def get_distance_map(modal_mask: torch.Tensor) -> torch.Tensor:
    device = modal_mask.device
    arr = modal_mask.detach().cpu().numpy()
    out = []
    for i in range(arr.shape[0]):
        m = (arr[i, 0] > 0.5).astype(np.uint8)
        if m.sum() == 0:
            out.append(np.zeros_like(m, dtype=np.float32))
            continue
        out.append(cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
    return torch.tensor(np.array(out), device=device).unsqueeze(1)


# ============================================================
# Scheduler (Exponential DA-UST)
# ============================================================
class ExponentialDAUSTScheduler:
    def __init__(self, num_train_timesteps=1000, gamma=5.0, beta_min=0.0,
                 image_size=256):
        self.num_train_timesteps = num_train_timesteps
        self.gamma = gamma
        self.beta_min = beta_min
        self.d_global = math.sqrt(image_size**2 + image_size**2)

        steps = torch.arange(num_train_timesteps + 1, dtype=torch.float32) / num_train_timesteps
        s = 0.008
        alphas = torch.cos(((steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        self.ust_alphas_cumprod = (alphas / alphas[0])[1:]

        self.gaussian_betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.gaussian_alphas_cumprod = torch.cumprod(1 - self.gaussian_betas, dim=0)

    def _compute_p_keep(self, bar_beta_t, distance_map):
        d_norm = torch.clamp(distance_map / self.d_global, 0.0, 1.0)
        g_d = 1.0 + self.gamma * (d_norm ** 2)
        return torch.clamp(torch.pow(bar_beta_t, g_d), min=self.beta_min)

    def add_noise(self, amodal_mask, distance_map, timesteps):
        device = amodal_mask.device
        self.ust_alphas_cumprod = self.ust_alphas_cumprod.to(device)
        self.gaussian_alphas_cumprod = self.gaussian_alphas_cumprod.to(device)

        bar_beta_t = self.ust_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        gauss_ab = self.gaussian_alphas_cumprod[timesteps].view(-1, 1, 1, 1)

        p_keep = self._compute_p_keep(bar_beta_t, distance_map)
        rand_t = torch.rand_like(amodal_mask)
        eroded = (rand_t < p_keep).float()
        modal = (distance_map == 0).float()
        x_prime = torch.where(modal == 1, amodal_mask, eroded * amodal_mask)

        eps = torch.randn_like(x_prime)
        return torch.sqrt(gauss_ab) * x_prime + torch.sqrt(1 - gauss_ab) * eps


# ============================================================
# PCN
# ============================================================
class ZOE(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.norm = nn.GroupNorm(1, embed_dim)
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x_t):
        return self.norm(self.proj(x_t)) * self.gate


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half = self.embed_dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class TimeTokenConcatenation(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.time_token_proj = nn.Sequential(nn.Linear(256, channels), nn.SiLU())
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads,
                                                batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(),
                                  nn.Linear(channels * 2, channels))

    def forward(self, F_i, t_emb):
        B, C, H, W = F_i.shape
        tokens = F_i.flatten(2).transpose(1, 2)
        time_token = self.time_token_proj(t_emb).unsqueeze(1)
        seq = torch.cat([time_token, tokens], dim=1)
        seq = seq + self.self_attn(self.norm1(seq), self.norm1(seq), self.norm1(seq))[0]
        seq = seq + self.ffn(self.norm2(seq))
        return seq[:, 1:, :].transpose(1, 2).reshape(B, C, H, W)


class PCNExtractor(nn.Module):
    def __init__(self, model_name='pvt_v2_b4', pretrained=True, embed_dim=768,
                 use_time_token_concat=True):
        super().__init__()
        self.use_time_token_concat = use_time_token_concat
        self.conv_c = nn.Sequential(
            nn.Conv2d(4, 3, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(3))

        self.pvt = timm.create_model(model_name, pretrained=pretrained,
                                      features_only=True)
        self.feature_channels = self.pvt.feature_info.channels()
        self.zoe = ZOE(1, self.feature_channels[0], 4)

        embed_layer = None
        for m in self.pvt.modules():
            if m.__class__.__name__ in ['OverlapPatchEmbed', 'PatchEmbed']:
                embed_layer = m
                break
        if embed_layer is None:
            raise AttributeError("Không tìm thấy Patch Embedding.")
        embed_layer.register_forward_hook(self._zoe_hook)

        self.time_embed = TimeEmbedding(256)
        self.time_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(256, c), nn.SiLU()) for c in self.feature_channels
        ])
        self.TTC_STAGE_INDICES = [2, 3]
        if self.use_time_token_concat:
            self.time_token_layers = nn.ModuleDict({
                str(i): TimeTokenConcatenation(self.feature_channels[i])
                for i in self.TTC_STAGE_INDICES
            })
        self.hf_proj = nn.Conv2d(self.feature_channels[-1], embed_dim, kernel_size=1)
        self._current_zoe = None

    def _zoe_hook(self, module, inputs, outputs):
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        z = self._current_zoe
        C = z.shape[1]
        if x.dim() == 4:
            x = x + z if x.shape[1] == C else x + z.permute(0, 2, 3, 1)
        elif x.dim() == 3:
            x = x + z.flatten(2).transpose(1, 2)
        return (x, *outputs[1:]) if isinstance(outputs, tuple) else x

    def forward(self, I, M_v, x_t, t, use_hf=False):
        X = self.conv_c(torch.cat([I, M_v], dim=1))
        self._current_zoe = self.zoe(x_t)
        feats = self.pvt(X)
        t_emb = self.time_embed(t)
        fused = []
        for i, F_ in enumerate(feats):
            if self.use_time_token_concat and i in self.TTC_STAGE_INDICES:
                fused.append(self.time_token_layers[str(i)](F_, t_emb))
            else:
                fused.append(F_ + self.time_projs[i](t_emb).unsqueeze(-1).unsqueeze(-1))
        if use_hf:
            hf = self.hf_proj(fused[-1])
            B_hf, C_hf, H_hf, W_hf = hf.shape
            return hf.view(B_hf, C_hf, -1).permute(0, 2, 1)
        return fused


# ============================================================
# DN v2 — ChannelAttention PHẢI khớp train (pool riêng, fc từ conv trở đi)
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class PyramidAttention(nn.Module):
    def __init__(self, channels, pool_sizes=(3, 5, 7), reduced_channels=None):
        super().__init__()
        self.pool_sizes = pool_sizes
        r = reduced_channels or max(channels // 4, 16)
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, r, 1), nn.BatchNorm2d(r), nn.ReLU(inplace=True))
            for _ in pool_sizes
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(r * len(pool_sizes), 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        feats = []
        for k, conv in zip(self.pool_sizes, self.scale_convs):
            feats.append(conv(F.avg_pool2d(x, k, 1, k // 2)))
        return self.fuse(torch.cat(feats, dim=1))


class OcclusionAwareContextRefinement(nn.Module):
    def __init__(self, channels, local_dilations=(1, 2), global_dilations=(3, 4)):
        super().__init__()
        bc = max(channels // 2, 32)

        def conv(ic, oc, d):
            return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=d, dilation=d),
                                  nn.BatchNorm2d(oc), nn.ReLU(inplace=True))

        self.local_branch = nn.Sequential(
            conv(channels, bc, local_dilations[0]),
            conv(bc, bc, local_dilations[1]), ChannelAttention(bc))
        self.global_branch = nn.Sequential(
            conv(channels, bc, global_dilations[0]),
            conv(bc, bc, global_dilations[1]), ChannelAttention(bc))
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, bc, 1),
            nn.BatchNorm2d(bc), nn.ReLU(inplace=True), ChannelAttention(bc))
        self.fuse = nn.Sequential(
            nn.Conv2d(bc * 3, channels, 1), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.pyramid_attn = PyramidAttention(channels)

    def forward(self, F4):
        H, W = F4.shape[-2:]
        lo = self.local_branch(F4)
        go = self.global_branch(F4)
        po = F.interpolate(self.pool_branch(F4), (H, W), mode='bilinear',
                            align_corners=False)
        fused = self.fuse(torch.cat([lo, go, po], dim=1))
        return F4 + fused * self.pyramid_attn(fused)


class LocalEmphasis(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list])

    def forward(self, features):
        size = features[0].shape[2:]
        out = []
        for i, f in enumerate(features):
            p = self.projs[i](f)
            if i > 0:
                p = F.interpolate(p, size=size, mode='bilinear', align_corners=False)
            out.append(p)
        return out


class AdaptiveFeatureGate(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, A2, F_up1):
        cat = torch.cat([A2, F_up1], dim=1)
        gate = torch.sigmoid(self.conv2(F.relu(self.conv1(cat))))
        return gate * A2 + (1.0 - gate) * F_up1


class UNetDenoisingNetwork(nn.Module):
    def __init__(self, pcn_channels=[64, 128, 320, 512], fuse_channels=256,
                 use_ocr=True):
        super().__init__()
        self.use_ocr = use_ocr
        if self.use_ocr:
            self.ocr_module = OcclusionAwareContextRefinement(pcn_channels[-1])
        self.le_module = LocalEmphasis(pcn_channels, fuse_channels)
        self.conv_A3 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1),
            nn.ReLU(inplace=True))
        self.conv_A2 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1),
            nn.ReLU(inplace=True))
        self.afg_module = AdaptiveFeatureGate(fuse_channels)

        self.enc1 = nn.Conv2d(2, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bottleneck = nn.Conv2d(256 + fuse_channels, 256, 3, padding=1)
        self.dec1_up = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec1_fuse = nn.Conv2d(128 + 128, 128, 3, padding=1)
        self.dec2_up = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2_fuse = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)

        self.time_embed = TimeEmbedding(256)
        ch = {"enc1": 64, "enc2": 128, "enc3": 256, "bottleneck": 256,
              "dec1": 128, "dec2": 64}
        self.time_projs = nn.ModuleDict({
            n: nn.Sequential(nn.Linear(256, c), nn.SiLU()) for n, c in ch.items()
        })

    def forward(self, x_t, t, pcn_features, M_v):
        F1, F2, F3, F4 = pcn_features
        if self.use_ocr:
            F4 = self.ocr_module(F4)
        Fu1, Fu2, Fu3, Fu4 = self.le_module([F1, F2, F3, F4])
        A3 = self.conv_A3(torch.cat([Fu4, Fu3], dim=1))
        A2 = self.conv_A2(torch.cat([A3, Fu2], dim=1))
        A1 = self.afg_module(A2, Fu1)

        t_emb = self.time_embed(t)

        def inj(feat, name):
            return feat + self.time_projs[name](t_emb).unsqueeze(-1).unsqueeze(-1)

        e1 = F.relu(inj(self.enc1(torch.cat([x_t, M_v], dim=1)), "enc1"))
        e2 = F.relu(inj(self.enc2(e1), "enc2"))
        e3 = F.relu(inj(self.enc3(e2), "enc3"))
        b = F.relu(inj(self.bottleneck(torch.cat([e3, A1], dim=1)), "bottleneck"))
        d1 = F.relu(self.dec1_fuse(torch.cat([F.relu(inj(self.dec1_up(b), "dec1")), e2], dim=1)))
        d2 = F.relu(self.dec2_fuse(torch.cat([F.relu(inj(self.dec2_up(d1), "dec2")), e1], dim=1)))
        return self.final_conv(d2)


# ============================================================
# Metrics — thống nhất: invisible hợp lệ khi gt_inv_area > 0
# ============================================================
def compute_metrics(pred_mask, gt_amodal, gt_modal, threshold=0.5):
    pred = (pred_mask > threshold).to(torch.int64)
    gt_a = gt_amodal.to(torch.int64)
    gt_m = gt_modal.to(torch.int64)

    inter = ((pred == 1) & (gt_a == 1)).sum(dim=(1, 2, 3))
    union = ((pred == 1) | (gt_a == 1)).sum(dim=(1, 2, 3))
    iou = inter.float() / (union.float() + 1e-6)

    gt_inv = (gt_a == 1) & (gt_m == 0)
    pred_inv = (pred == 1) & (gt_m == 0)
    inter_inv = (pred_inv & gt_inv).sum(dim=(1, 2, 3))
    union_inv = (pred_inv | gt_inv).sum(dim=(1, 2, 3))
    inv_iou = inter_inv.float() / (union_inv.float() + 1e-6)

    # Thống nhất: bất kỳ mẫu nào có invisible GT đều hợp lệ
    valid_inv = gt_inv.sum(dim=(1, 2, 3)) > 0
    return iou, inv_iou, valid_inv


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================
# VGPE formulas
# ============================================================
def formula_A_sum(all_soft, all_binary):
    A = torch.stack(all_binary, 0).mean(0)
    B = torch.stack(all_soft, 0).mean(0)
    return ((A + 0.5 * B) > 0.5).float()


def formula_B_literal(all_soft, all_binary):
    A = torch.stack(all_binary, 0).mean(0)
    B = torch.stack(all_soft, 0).mean(0)
    return ((torch.abs(A + 0.5) * B) > 0.5).float()


def formula_C_majority(all_soft, all_binary):
    A = torch.stack(all_binary, 0).mean(0)
    B = torch.stack(all_soft, 0).mean(0)
    return ((torch.floor(A + 0.5) * B) > 0.5).float()


def formula_D_gating(all_soft, all_binary):
    A = torch.stack(all_binary, 0).mean(0)
    B = torch.stack(all_soft, 0).mean(0)
    return ((2 * torch.abs(A - 0.5) * B) > 0.5).float()


FORMULAS = {
    "A (Sum)":      formula_A_sum,
    "B (Literal)":  formula_B_literal,
    "C (Majority)": formula_C_majority,
    "D (Gating)":   formula_D_gating,
}
FORMULA_ORDER = list(FORMULAS.keys())


# ============================================================
# Globals
# ============================================================
DEVICE = torch.device("cpu")
pcn = dn = scheduler = gauss_ac = None
val_loader = None
OUTPUT_DIR = "./eval_results"
INFERENCE_STEPS = DEFAULT_STEPS
MAX_EVAL_SAMPLES = None


# ============================================================
# Sampling — cố định noise theo seed để so sánh 3 configs fair
# ============================================================
@torch.no_grad()
def get_ensemble_components(I, M_v, distance_map, seed, use_correct_timing=True,
                            use_ust_reverse=True):
    """
    seed: cố định per-(batch, config)? Không — cố định per-batch để so sánh fair.
    Bên trong hàm tự reset RNG = seed + batch_index (caller truyền).
    """
    # fork_rng chỉ ảnh hưởng CUDA/CPU RNG trong scope — không làm nhiễu training
    fork_devices = []
    if DEVICE.type == "cuda":
        cur_idx = DEVICE.index if DEVICE.index is not None else torch.cuda.current_device()
        fork_devices = [cur_idx]

    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        B = M_v.shape[0]
        timesteps = np.linspace(TIMESTEPS - 1, 0, INFERENCE_STEPS, dtype=int).tolist()

        a_bar_T = gauss_ac[TIMESTEPS - 1].view(-1, 1, 1, 1)
        x_t = torch.sqrt(a_bar_T) * M_v + torch.sqrt(1 - a_bar_T) * torch.randn_like(M_v)

        all_soft, all_binary = [], []
        for idx, t_val in enumerate(timesteps):
            t_tensor = torch.full((B,), t_val, device=DEVICE, dtype=torch.long)
            feats = pcn(I, M_v, x_t, t_tensor)
            x0_raw = torch.sigmoid(dn(x_t, t_tensor, feats, M_v))

            if use_correct_timing:
                apply = (t_val >= TIMESTEPS // 2)
            else:
                apply = (idx >= len(timesteps) // 2)

            x0_pred = x0_raw * (1.0 - M_v) + M_v if apply else x0_raw

            all_soft.append(x0_pred)
            all_binary.append((x0_pred > 0.5).float())

            if idx < len(timesteps) - 1:
                t_prev = timesteps[idx + 1]
                if use_ust_reverse:
                    tp = torch.full((B,), t_prev, device=DEVICE, dtype=torch.long)
                    x_t = scheduler.add_noise(x0_pred, distance_map, tp)
                else:
                    a_t = gauss_ac[t_val].view(-1, 1, 1, 1)
                    a_p = gauss_ac[t_prev].view(-1, 1, 1, 1)
                    eps = (x_t - torch.sqrt(a_t) * x0_pred) / torch.sqrt(1.0 - a_t + 1e-8)
                    x_t = torch.sqrt(a_p) * x0_pred + torch.sqrt(1.0 - a_p) * eps

        return all_soft, all_binary


def run_one_config(use_correct_timing, use_ust_reverse, label, seed, save_visual=False):
    print(f"\n{'='*90}")
    print(f"⚙️  {label}  |  timing={use_correct_timing}  ust_reverse={use_ust_reverse}  "
          f"seed={seed}")
    print(f"{'='*90}")

    iou_m = {n: AverageMeter() for n in FORMULA_ORDER}
    inv_m = {n: AverageMeter() for n in FORMULA_ORDER}
    per_sample_inv = {n: [] for n in FORMULA_ORDER}

    n_seen = 0
    pbar = tqdm(val_loader, desc=f"[{label}]")
    for batch_idx, (I, M_v, M_a, _) in enumerate(pbar):
        if MAX_EVAL_SAMPLES is not None and n_seen >= MAX_EVAL_SAMPLES:
            break

        I, M_v, M_a = I.to(DEVICE), M_v.to(DEVICE), M_a.to(DEVICE)

        dm = get_distance_map(M_v)
        batch_seed = seed + batch_idx    # cố định theo batch, không theo config
        all_soft, all_binary = get_ensemble_components(
            I, M_v, dm, seed=batch_seed,
            use_correct_timing=use_correct_timing,
            use_ust_reverse=use_ust_reverse,
        )

        if save_visual and n_seen == 0:
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            pred_d = formula_D_gating(all_soft, all_binary)
            for i in range(min(4, I.shape[0])):
                img = (I[i].cpu().permute(1, 2, 0).numpy()
                       * np.array([0.229, 0.224, 0.225])
                       + np.array([0.485, 0.456, 0.406]))
                axes[0, i].imshow(np.clip(img, 0, 1)); axes[0, i].set_title(f"Image {i}")
                axes[1, i].imshow(M_v[i, 0].cpu(), cmap='gray'); axes[1, i].set_title("M_v")
                axes[2, i].imshow(pred_d[i, 0].cpu(), cmap='gray'); axes[2, i].set_title("Pred (D)")
                axes[3, i].imshow(M_a[i, 0].cpu(), cmap='gray'); axes[3, i].set_title("M_a")
                for ax in axes[:, i]:
                    ax.axis('off')
            plt.tight_layout()
            out = OUTPUT_DIR / f"vgpe_{label.replace(' ', '_')}.png"
            plt.savefig(out, dpi=120)
            plt.close()
            print(f"📸 Saved: {out}")

        for name, fn in FORMULAS.items():
            pred_mask = fn(all_soft, all_binary)
            iou, inv_iou, valid_inv = compute_metrics(pred_mask, M_a, M_v)

            iou_m[name].update(iou.mean().item(), I.size(0))
            if valid_inv.any():
                inv_m[name].update(inv_iou[valid_inv].mean().item(),
                                    int(valid_inv.sum().item()))
                for i in range(I.size(0)):
                    if valid_inv[i]:
                        per_sample_inv[name].append(inv_iou[i].item())

        n_seen += I.size(0)

    print(f"\n{'Formula':<16} | {'mIoU_full':<11} | {'Inv-mIoU':<11} | "
          f"{'Std':<8} | p25/p50/p75")
    print("-" * 90)
    for name in FORMULA_ORDER:
        vals = np.array(per_sample_inv[name])
        if len(vals) == 0:
            print(f"{name:<16} | (no valid invisible samples)")
            continue
        p25, p50, p75 = np.percentile(vals, [25, 50, 75]) * 100
        print(f"{name:<16} | {iou_m[name].avg*100:6.2f}%     | "
              f"{inv_m[name].avg*100:6.2f}%     | "
              f"{vals.std()*100:6.2f}%  | {p25:.1f}/{p50:.1f}/{p75:.1f}")

    return {
        "per_sample_inv": per_sample_inv,
        "iou_avg": {n: iou_m[n].avg for n in FORMULA_ORDER},
        "inv_avg": {n: inv_m[n].avg for n in FORMULA_ORDER},
    }


# ============================================================
# Main
# ============================================================
def run_test():
    global val_loader, pcn, dn, scheduler, gauss_ac, DEVICE
    global MAX_EVAL_SAMPLES, INFERENCE_STEPS, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Evaluate CondDiff-AMO checkpoint.")
    parser.add_argument("--checkpoint", "-c", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--coco-img-dir", type=str, default=str(DEFAULT_COCO_IMG_DIR))
    parser.add_argument("--cocoa-val-json", type=str, default=str(DEFAULT_COCOA_VAL_JSON))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="<= 0 = chạy toàn bộ validation")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=2024,
                        help="Base seed cho sampling (cố định giữa 3 configs)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.data_root:
        root = _Path(args.data_root).resolve()
        args.coco_img_dir   = str(root / "coco_amodal" / "cocoa_images_extracted")
        args.cocoa_val_json = str(root / "coco_amodal" / "COCO_amodal_val2014.json")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint không tồn tại: {args.checkpoint}")
    if not os.path.isfile(args.cocoa_val_json):
        raise FileNotFoundError(f"JSON không tồn tại: {args.cocoa_val_json}")
    if not os.path.isdir(args.coco_img_dir):
        raise NotADirectoryError(f"Thư mục ảnh không tồn tại: {args.coco_img_dir}")

    MAX_EVAL_SAMPLES = (None if args.max_samples <= 0 else args.max_samples)
    INFERENCE_STEPS = args.steps
    OUTPUT_DIR = _Path(args.output_dir).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        DEVICE = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng.")
        DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}"
                              if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {DEVICE}")

    val_ds = AmodalDataset(
        data_dir=args.cocoa_val_json, mode="cocoa_val",
        image_size=IMAGE_SIZE, coco_img_dir=args.coco_img_dir)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(DEVICE.type == "cuda"))

    print("🔧 Khởi tạo PCN + DN v2 (pretrained=False, sẽ load từ checkpoint)...")
    pcn = PCNExtractor(pretrained=False).to(DEVICE)
    dn = UNetDenoisingNetwork(pcn_channels=pcn.feature_channels,
                               fuse_channels=256).to(DEVICE)

    try:
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)

    if "pcn_state_dict" not in ckpt or "dn_state_dict" not in ckpt:
        raise KeyError("Checkpoint thiếu pcn_state_dict / dn_state_dict.")
    # strict=True (default): phải load đủ weight, không dùng strict=False để né lỗi
    pcn.load_state_dict(ckpt["pcn_state_dict"])
    dn.load_state_dict(ckpt["dn_state_dict"])
    print(f"✅ Loaded: epoch={ckpt.get('epoch', '?')} "
          f"best_miou_inv={ckpt.get('best_miou_inv', '?')}")

    if not ckpt.get("use_ust_forward", True):
        print("⚠️ Checkpoint no-UST — UST-reverse không hợp lệ với checkpoint này.")

    pcn.eval(); dn.eval()

    scheduler = ExponentialDAUSTScheduler(
        num_train_timesteps=TIMESTEPS, gamma=5.0, beta_min=0.0)
    scheduler.gaussian_betas = torch.linspace(1e-5, 2e-3, TIMESTEPS)
    scheduler.gaussian_alphas_cumprod = torch.cumprod(1 - scheduler.gaussian_betas, dim=0)
    scheduler.ust_alphas_cumprod = scheduler.ust_alphas_cumprod.to(DEVICE)
    scheduler.gaussian_alphas_cumprod = scheduler.gaussian_alphas_cumprod.to(DEVICE)
    gauss_ac = scheduler.gaussian_alphas_cumprod

    n_eval = "ALL" if MAX_EVAL_SAMPLES is None else MAX_EVAL_SAMPLES
    print(f"📊 {len(val_ds)} objects | eval {n_eval} | steps {INFERENCE_STEPS} | "
          f"seed {args.seed}")

    # 3 configs dùng cùng seed
    configs = [
        ("v3_baseline", False, False, "v3_baseline (bug timing + DDIM)", False),
        ("fix1_only",   True,  False, "fix1_only (fixed timing + DDIM)", False),
        ("fix1_fix2",   True,  True,  "fix1_fix2 (fixed timing + UST-reverse)", True),
    ]
    results = {k: run_one_config(t, u, lbl, args.seed, save_visual=v)
               for k, t, u, lbl, v in configs}

    print(f"\n{'='*90}\nSO SÁNH 3 CONFIGS — Formula D (Gating)\n{'='*90}")
    v3_inv = results["v3_baseline"]["inv_avg"][FORMULA_ORDER[3]] * 100
    print(f"{'Config':<20} | {'mIoU_full':<11} | {'Inv-mIoU':<11} | Δ Inv vs v3")
    for k, *_ in configs:
        full = results[k]["iou_avg"][FORMULA_ORDER[3]] * 100
        inv = results[k]["inv_avg"][FORMULA_ORDER[3]] * 100
        print(f"{k:<20} | {full:6.2f}%     | {inv:6.2f}%     | {inv-v3_inv:+6.2f}%")

    best = max(results, key=lambda c: results[c]["inv_avg"][FORMULA_ORDER[3]])
    print(f"\n🏆 Best config: {best}")
    print(f"   Inv-mIoU (D) = {results[best]['inv_avg'][FORMULA_ORDER[3]]*100:.2f}%")

    del pcn, dn, scheduler, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return results


if __name__ == "__main__":
    try:
        run_test()
    except KeyboardInterrupt:
        print("\n⛔ Dừng bởi người dùng.")
        sys.exit(130)
    except Exception:
        print("\n[FATAL]", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)