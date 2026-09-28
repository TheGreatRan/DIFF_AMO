#!/usr/bin/env python3
"""
CondDiff-AMO — SINGLE-FILE self-contained training script.

MỤC TIÊU:
    Copy 1 file này sang server trường là chạy được.
    KHÔNG cần config.py riêng, KHÔNG cần src/ folder.
    Chỉ cần chỉnh CONFIG ở section ngay dưới đây, rồi:  python train_all_in_one.py

REPRODUCE:
    Bản này reproduce chính xác notebook train(1).ipynb (v2) về:
        - DA-UST exponential gamma=5, beta_min=0
        - Gaussian schedule (1e-5, 2e-3)
        - DN v2 (UNetDenoisingNetwork) nhận (x_t, t, feats, M_v)
        - Fixed-corruption validation t=T-1, VAL_SEED=12345
        - sqrt(u) timestep sampling
        - weight_decay=5e-2

STABILITY MODIFICATIONS (không có trong notebook gốc, CÓ THỂ TẮT):
    - Grad clipping 1.0  → set GRAD_CLIP_NORM = None để tắt
    - Deterministic validation fork_rng → set FORK_RNG_VALIDATION = False để tắt
    - dataset filtering (filter_degenerate/fully_occluded) → default False = giữ nguyên notebook

USAGE:
    python train_all_in_one.py
    python train_all_in_one.py --data-root /mnt/ssd/data
    python train_all_in_one.py --resume /path/to/best_miou_ckpt.pth
    python train_all_in_one.py --max-hours 6 --epochs 20
"""

# ============================================================================
# ============================================================================
#                              C O N F I G
#                CHỈNH SỬA Ở ĐÂY, KHÔNG CẦN SỬA GÌ KHÁC
# ============================================================================
# ============================================================================
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent

# ---- 1. EXPERIMENT ----
EXPERIMENT_MODE = "zero-shot"      # "supervised" | "zero-shot"
STRATEGY_NAME   = "exponential"     # "baseline" | "clamped" | "exponential"
DN_VARIANT      = "new"             # "old" (DNv1) | "new" (DNv2 U-Net)
SEED            = 42

# ---- 2. HYPERPARAMETERS ----
NUM_EPOCHS      = 40
LEARNING_RATE   = 1e-4
BATCH_SIZE      = 16
IMAGE_SIZE      = (256, 256)
TIMESTEPS       = 1000
MAX_TRAIN_HOURS = 120
NUM_WORKERS     = 2

# ---- 3. LOSS ----
OCCLUSION_LOSS_WEIGHT     = 10.0
EDGE_LOSS_WEIGHT          = 0.0
EDGE_LOSS_BOUNDARY_SOURCE = "modal"
EARLY_STOP_PATIENCE       = 0

# ---- 4. DA-UST ----
DA_UST_GAMMA    = 5.0
DA_UST_BETA_MIN = 0.0

# ---- 5. GAUSSIAN SCHEDULE ----
GAUSS_BETA_START = 1e-5
GAUSS_BETA_END   = 2e-3

# ---- 6. VALIDATION ----
VAL_SEED              = 12345
FORK_RNG_VALIDATION   = True    # True → deterministic; False → random mỗi epoch

# ---- 7. STABILITY (không có trong notebook gốc) ----
GRAD_CLIP_NORM = None            # None để tắt (khớp notebook gốc)

# ---- 8. AUGMENTATION (per-type flags — mirror notebook v2) ----
# QUAN TRỌNG: Đây là các flag THỰC SỰ có tác dụng vì AmodalDataset
# trong file này đã được mở rộng để nhận từng flag riêng.
AUGMENT_HFLIP               = True
AUGMENT_SCALE_CROP          = False
AUGMENT_ROTATE              = False
AUGMENT_COLOR_JITTER        = False
AUGMENT_SYNTHETIC_OCCLUSION = False

# ---- 9. DATASET FILTERING (default False = khớp notebook gốc) ----
FILTER_DEGENERATE     = False
FILTER_FULLY_OCCLUDED = False
MIN_AREA_PX           = 4

# ---- 10. PATHS ----
DATA_ROOT = _PROJECT_ROOT / "data"

COCO_IMG_DIR     = str(DATA_ROOT / "coco_amodal" / "cocoa_images_extracted")
PIX2GESTALT_DIR  = str(DATA_ROOT / "pix2gestalt" / "pix2gestalt_occlusions_release")
COCOA_TRAIN_JSON = str(DATA_ROOT / "coco_amodal" / "COCO_amodal_train2014.json")
COCOA_VAL_JSON   = str(DATA_ROOT / "coco_amodal" / "COCO_amodal_val2014.json")

SAVE_DIR = str(_PROJECT_ROOT / "checkpoints")

# ---- 11. RESUME ----
RESUME_FROM_EXTERNAL = None     # None = train từ đầu


# ============================================================================
# ============================================================================
#                            I M P O R T S
# ============================================================================
# ============================================================================
import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

PROJECT_ROOT = _PROJECT_ROOT


# ============================================================================
# ============================================================================
#                     A M O D A L   D A T A S E T
#           (đã mở rộng với per-type augmentation flags)
# ============================================================================
# ============================================================================
class AmodalDataset(Dataset):
    """
    Amodal segmentation dataset.

    BẢN MỞ RỘNG: nhận từng flag augmentation riêng (mirror notebook v2),
    thay vì chỉ 1 cờ `augment` như bản gốc trong repo.

    Backward compat: nếu chỉ truyền `augment=X`, thì 5 flag con kế thừa X.
    """

    def __init__(
        self,
        data_dir,
        mode="toy",
        image_size=(256, 256),
        coco_img_dir=None,
        # Master switch (giữ để backward compat)
        augment=None,
        # Per-type overrides (None = kế thừa `augment`)
        augment_hflip=None,
        augment_scale_crop=None,
        augment_rotate=None,
        augment_color_jitter=None,
        augment_synthetic_occlusion=None,
        # Optional dataset filtering (default False = reproduce notebook gốc)
        filter_degenerate=False,
        filter_fully_occluded=False,
        min_area_px=4,
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.coco_img_dir = coco_img_dir
        self.samples = []

        base_augment = (
            mode in ("pix2gestalt", "cocoa_train") if augment is None else augment
        )
        self.aug_hflip = base_augment if augment_hflip is None else augment_hflip
        self.aug_scale_crop = base_augment if augment_scale_crop is None else augment_scale_crop
        self.aug_rotate = base_augment if augment_rotate is None else augment_rotate
        self.aug_color_jitter = base_augment if augment_color_jitter is None else augment_color_jitter
        self.aug_synthetic_occlusion = (
            base_augment if augment_synthetic_occlusion is None
            else augment_synthetic_occlusion
        )
        self.augment = base_augment  # legacy attribute

        # ---- TOY ----
        if self.mode == "toy":
            print("🚀 Đang khởi tạo Toy Dataset...")
            self.num_samples = 100

        # ---- PIX2GESTALT ----
        elif self.mode == "pix2gestalt":
            print("⏳ Đang quét và đối chiếu chéo dữ liệu Pix2Gestalt ...")
            self.occlusion_dir = os.path.join(data_dir, 'occlusion')
            self.visible_mask_dir = os.path.join(data_dir, 'visible_object_mask')
            self.whole_mask_dir = os.path.join(data_dir, 'whole_mask')

            occ_ids = {f.split('_')[0] for f in os.listdir(self.occlusion_dir) if f.endswith('.png')}
            vis_ids = {f.split('_')[0] for f in os.listdir(self.visible_mask_dir) if f.endswith('.png')}
            whole_ids = {f.split('_')[0] for f in os.listdir(self.whole_mask_dir) if f.endswith('.png')}

            valid_ids = occ_ids & vis_ids & whole_ids
            self.base_ids = sorted(list(valid_ids))
            self.num_samples = len(self.base_ids)
            print(f"✅ Đã tìm thấy {self.num_samples} mẫu hợp lệ.")

        # ---- COCOA ----
        elif self.mode.startswith("cocoa"):
            print(f"⏳ Đang phân tích đồ thị JSON của {self.mode.upper()} ...")
            if not self.coco_img_dir:
                raise ValueError("Chế độ COCOA yêu cầu phải truyền 'coco_img_dir'.")

            print("💾 Đang lập chỉ mục thư mục ảnh phẳng...")
            self.available_images = set(os.listdir(self.coco_img_dir))

            with open(self.data_dir, 'r') as f:
                cocoa_data = json.load(f)

            skipped_images_count = 0
            skipped_objects_count = 0
            skipped_degenerate_count = 0
            skipped_fully_occluded_count = 0
            candidate_samples = []

            for ann in cocoa_data['annotations']:
                img_filename = ann['url'].split('/')[-1]
                if img_filename not in self.available_images:
                    skipped_images_count += 1
                    skipped_objects_count += len(
                        [r for r in ann['regions'] if r.get('isStuff', 0) != 1]
                    )
                    continue

                depth_constraints = ann.get('depth_constraint', '')
                occluders_of = {}
                if depth_constraints:
                    for pair in depth_constraints.split(','):
                        if '-' in pair:
                            front, back = pair.split('-')
                            front, back = int(front), int(back)
                            occluders_of.setdefault(back, []).append(front)

                regions = ann['regions']
                for region in regions:
                    if region.get('isStuff', 0) == 1:
                        continue
                    region_order = region.get('order')
                    if region_order is None:
                        continue

                    seg = region.get('segmentation')
                    if not seg or len(seg) == 0:
                        skipped_degenerate_count += 1
                        continue

                    candidate_samples.append({
                        'img_filename': img_filename,
                        'region': region,
                        'region_order': region_order,
                        'all_regions': regions,
                        'occluders': occluders_of.get(region_order, []),
                    })

            # Optional filtering (default OFF = reproduce notebook gốc)
            if filter_degenerate or filter_fully_occluded:
                for s in candidate_samples:
                    try:
                        amodal_area, visible_area = self._fast_areas(
                            s['region']['segmentation'],
                            self._collect_occluder_polys(s['occluders'], s['all_regions']),
                        )
                    except Exception:
                        skipped_degenerate_count += 1
                        continue
                    if filter_degenerate and amodal_area < min_area_px:
                        skipped_degenerate_count += 1
                        continue
                    if filter_fully_occluded and visible_area < min_area_px:
                        skipped_fully_occluded_count += 1
                        continue
                    self.samples.append(s)
            else:
                self.samples = candidate_samples

            self.num_samples = len(self.samples)
            print(f"✅ Đã giải mã thành công {self.num_samples} vật thể từ COCOA.")
            if skipped_images_count > 0:
                print(f"⚠️ Bỏ qua {skipped_images_count} ảnh thiếu ({skipped_objects_count} vật thể).")
            if skipped_degenerate_count > 0:
                print(f"⚠️ Lọc {skipped_degenerate_count} region degenerate.")
            if filter_fully_occluded and skipped_fully_occluded_count > 0:
                print(f"⚠️ Lọc {skipped_fully_occluded_count} region che hoàn toàn.")
        else:
            raise ValueError(f"mode='{self.mode}' không được hỗ trợ!")

    # ---- fast polygon area (không decode ảnh) ----
    @staticmethod
    def _collect_occluder_polys(occluder_orders, all_regions):
        polys = []
        for occ_order in occluder_orders:
            for r in all_regions:
                if r.get('order') == occ_order:
                    seg = r.get('segmentation')
                    if seg:
                        polys.append(seg)
                    break
        return polys

    @staticmethod
    def _poly_to_points_list(poly):
        if isinstance(poly[0], list):
            return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in poly]
        return [np.array(poly, dtype=np.float32).reshape(-1, 2)]

    def _fast_areas(self, amodal_poly, occluder_polys):
        amodal_pts_list = self._poly_to_points_list(amodal_poly)
        occ_pts_lists = [self._poly_to_points_list(p) for p in occluder_polys]
        all_pts = np.concatenate(
            amodal_pts_list + [p for lst in occ_pts_lists for p in lst], axis=0
        )
        min_xy = all_pts.min(axis=0) - 2
        max_xy = all_pts.max(axis=0) + 2
        w = max(int(np.ceil(max_xy[0] - min_xy[0])), 1)
        h = max(int(np.ceil(max_xy[1] - min_xy[1])), 1)

        def rasterize(pts_list):
            mask = np.zeros((h, w), dtype=np.uint8)
            polys_int = [(pts - min_xy).astype(np.int32) for pts in pts_list]
            cv2.fillPoly(mask, polys_int, 1)
            return mask

        amodal_mask = rasterize(amodal_pts_list)
        occ_mask = np.zeros((h, w), dtype=np.uint8)
        for pts_list in occ_pts_lists:
            occ_mask = np.logical_or(occ_mask, rasterize(pts_list)).astype(np.uint8)
        visible_mask = amodal_mask.copy()
        visible_mask[occ_mask == 1] = 0
        return int(amodal_mask.sum()), int(visible_mask.sum())

    def __len__(self):
        return self.num_samples

    def _poly_to_mask(self, polygons, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        if isinstance(polygons[0], list):
            pts = [np.array(p, np.int32).reshape((-1, 2)) for p in polygons]
        else:
            pts = [np.array(polygons, np.int32).reshape((-1, 2))]
        cv2.fillPoly(mask, pts, 1)
        return mask

    # ---- per-type augmentation ----
    def _maybe_hflip(self, img, m_v, m_a):
        if self.aug_hflip and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, ...])
            m_v = np.ascontiguousarray(m_v[:, ::-1])
            m_a = np.ascontiguousarray(m_a[:, ::-1])
        return img, m_v, m_a

    def _maybe_scale_crop(self, img, m_v, m_a):
        if not self.aug_scale_crop:
            return img, m_v, m_a
        H, W = img.shape[:2]
        scale = np.random.uniform(0.85, 1.0)
        new_h, new_w = int(round(H * scale)), int(round(W * scale))
        if new_h >= H or new_w >= W or new_h <= 0 or new_w <= 0:
            return img, m_v, m_a
        top = np.random.randint(0, H - new_h + 1)
        left = np.random.randint(0, W - new_w + 1)
        img_c = img[top:top + new_h, left:left + new_w]
        m_v_c = m_v[top:top + new_h, left:left + new_w]
        m_a_c = m_a[top:top + new_h, left:left + new_w]
        img = cv2.resize(img_c, (W, H), interpolation=cv2.INTER_LINEAR)
        m_v = cv2.resize(m_v_c, (W, H), interpolation=cv2.INTER_NEAREST)
        m_a = cv2.resize(m_a_c, (W, H), interpolation=cv2.INTER_NEAREST)
        return img, m_v, m_a

    def _maybe_color_jitter(self, img):
        if not self.aug_color_jitter:
            return img
        img = img.astype(np.float32)
        brightness = np.random.uniform(0.85, 1.15)
        contrast = np.random.uniform(0.85, 1.15)
        mean_val = img.mean()
        img = (img - mean_val) * contrast + mean_val
        img = img * brightness
        return np.clip(img, 0, 255)

    def _maybe_rotate(self, img, m_v, m_a, max_angle=15):
        if not self.aug_rotate or np.random.rand() < 0.5:
            return img, m_v, m_a
        H, W = img.shape[:2]
        angle = np.random.uniform(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
        m_v = cv2.warpAffine(m_v, M, (W, H), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m_a = cv2.warpAffine(m_a, M, (W, H), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, m_v, m_a

    def _maybe_synthetic_occlusion(self, m_v, m_a, prob=0.4, max_patches=2,
                                    patch_scale_range=(0.05, 0.25)):
        if not self.aug_synthetic_occlusion or np.random.rand() > prob:
            return m_v
        H, W = m_v.shape
        ys, xs = np.where(m_v > 0)
        if len(ys) == 0:
            return m_v
        m_v = m_v.copy()
        n_patches = np.random.randint(1, max_patches + 1)
        for _ in range(n_patches):
            idx = np.random.randint(len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])
            patch_h = max(1, int(H * np.random.uniform(*patch_scale_range)))
            patch_w = max(1, int(W * np.random.uniform(*patch_scale_range)))
            y0, y1 = max(0, cy - patch_h // 2), min(H, cy + patch_h // 2)
            x0, x1 = max(0, cx - patch_w // 2), min(W, cx + patch_w // 2)
            m_v[y0:y1, x0:x1] = 0
        return m_v

    def _augment_all(self, img, m_v, m_a):
        img, m_v, m_a = self._maybe_hflip(img, m_v, m_a)
        img, m_v, m_a = self._maybe_scale_crop(img, m_v, m_a)
        img, m_v, m_a = self._maybe_rotate(img, m_v, m_a)
        img = self._maybe_color_jitter(img)
        m_v = self._maybe_synthetic_occlusion(m_v, m_a)
        return img, m_v, m_a

    def __getitem__(self, idx):
        if self.mode == "toy":
            return self._generate_toy_data()

        elif self.mode == "pix2gestalt":
            base_id = self.base_ids[idx]
            img_path = os.path.join(self.occlusion_dir, f"{base_id}_occlusion.png")
            m_v_path = os.path.join(self.visible_mask_dir, f"{base_id}_visible_mask.png")
            m_a_path = os.path.join(self.whole_mask_dir, f"{base_id}_whole_mask.png")

            img = cv2.imread(img_path)
            m_v = cv2.imread(m_v_path, cv2.IMREAD_GRAYSCALE)
            m_a = cv2.imread(m_a_path, cv2.IMREAD_GRAYSCALE)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            m_v = cv2.resize(m_v, self.image_size, interpolation=cv2.INTER_NEAREST)
            m_a = cv2.resize(m_a, self.image_size, interpolation=cv2.INTER_NEAREST)

            img, m_v, m_a = self._augment_all(img, m_v, m_a)

            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            m_v = (m_v > 127).astype(np.float32)
            m_a = (m_a > 127).astype(np.float32)

            img = np.transpose(img, (2, 0, 1))
            m_v = np.expand_dims(m_v, axis=0)
            m_a = np.expand_dims(m_a, axis=0)
            return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a)

        elif self.mode.startswith("cocoa"):
            sample = self.samples[idx]
            img_filename = sample['img_filename']
            img_path = os.path.join(self.coco_img_dir, img_filename)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img.shape[:2]

            target_poly = sample['region']['segmentation']
            amodal_mask = self._poly_to_mask(target_poly, h_orig, w_orig)

            occluder_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            all_regions = sample['all_regions']
            for occ_order in sample['occluders']:
                for r in all_regions:
                    if r.get('order') == occ_order:
                        occ_poly = r['segmentation']
                        occ_mask = self._poly_to_mask(occ_poly, h_orig, w_orig)
                        occluder_mask = np.logical_or(occluder_mask, occ_mask).astype(np.uint8)
                        break

            modal_mask = amodal_mask.copy()
            modal_mask[occluder_mask == 1] = 0

            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            m_v = cv2.resize(modal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            m_a = cv2.resize(amodal_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            img, m_v, m_a = self._augment_all(img, m_v, m_a)

            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            m_v = m_v.astype(np.float32)
            m_a = m_a.astype(np.float32)

            img = np.transpose(img, (2, 0, 1))
            m_v = np.expand_dims(m_v, axis=0)
            m_a = np.expand_dims(m_a, axis=0)
            return torch.tensor(img), torch.tensor(m_v), torch.tensor(m_a)


# ============================================================================
# ============================================================================
#                    G E O M E T R Y   /   L O S S   W E I G H T S
# ============================================================================
# ============================================================================
def get_distance_map(modal_mask: torch.Tensor) -> torch.Tensor:
    device = modal_mask.device
    masks_np = modal_mask.detach().cpu().numpy()
    dist_maps = []
    for i in range(masks_np.shape[0]):
        mask_single = (masks_np[i, 0] > 0.5).astype(np.uint8)
        if mask_single.sum() == 0:
            dist_maps.append(np.zeros_like(mask_single, dtype=np.float32))
            continue
        inverted = 1 - mask_single
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_maps.append(dist)
    return torch.tensor(np.array(dist_maps), device=device).unsqueeze(1)


def compute_occlusion_weight_map(amodal_mask, modal_mask,
                                  occlusion_weight=3.0, background_weight=1.0):
    occluded_region = amodal_mask * (1.0 - modal_mask)
    return background_weight + (occlusion_weight - background_weight) * occluded_region


# ============================================================================
# ============================================================================
#                    D A - U S T   S C H E D U L E R S
# ============================================================================
# ============================================================================
class BaseDAUSTScheduler:
    def __init__(self, num_train_timesteps=1000, sigma=5.0, beta_min=0.05,
                 gamma=5.0, image_size=256):
        self.num_train_timesteps = num_train_timesteps
        self.sigma = sigma
        self.beta_min = beta_min
        self.gamma = gamma
        self.image_size = image_size
        self.d_global = math.sqrt(image_size**2 + image_size**2)

        steps = torch.arange(num_train_timesteps + 1, dtype=torch.float32) / num_train_timesteps
        s = 0.008
        alphas = torch.cos(((steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        self.ust_alphas_cumprod = (alphas / alphas[0])[1:]

        self.gaussian_betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.gaussian_alphas_cumprod = torch.cumprod(1 - self.gaussian_betas, dim=0)

    def _compute_p_keep(self, bar_beta_t, distance_map):
        raise NotImplementedError

    def add_noise(self, amodal_mask, distance_map, timesteps, return_intermediate=False):
        device = amodal_mask.device
        self.ust_alphas_cumprod = self.ust_alphas_cumprod.to(device)
        self.gaussian_alphas_cumprod = self.gaussian_alphas_cumprod.to(device)

        bar_beta_t = self.ust_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        gauss_alpha_bar_t = self.gaussian_alphas_cumprod[timesteps].view(-1, 1, 1, 1)

        p_keep = self._compute_p_keep(bar_beta_t, distance_map)

        rand_tensor = torch.rand_like(amodal_mask)
        eroded_mask = (rand_tensor < p_keep).float()
        modal_mask = (distance_map == 0).float()
        x_prime_t = torch.where(modal_mask == 1, amodal_mask, eroded_mask * amodal_mask)

        epsilon = torch.randn_like(x_prime_t)
        x_t = (torch.sqrt(gauss_alpha_bar_t) * x_prime_t
               + torch.sqrt(1 - gauss_alpha_bar_t) * epsilon)

        if return_intermediate:
            return x_t, x_prime_t
        return x_t


class BaselineUSTScheduler(BaseDAUSTScheduler):
    def _compute_p_keep(self, bar_beta_t, distance_map):
        return bar_beta_t.expand_as(distance_map)


class ClampedDAUSTScheduler(BaseDAUSTScheduler):
    def _compute_p_keep(self, bar_beta_t, distance_map):
        d_norm = torch.clamp(distance_map / self.d_global, 0.0, 1.0)
        sigma_norm = 0.1
        spatial_decay = torch.exp(-(d_norm ** 2) / (2 * (sigma_norm ** 2)))
        clamped_decay = torch.clamp(spatial_decay, min=self.beta_min)
        return bar_beta_t * clamped_decay


class ExponentialDAUSTScheduler(BaseDAUSTScheduler):
    def _compute_p_keep(self, bar_beta_t, distance_map):
        d_norm = torch.clamp(distance_map / self.d_global, 0.0, 1.0)
        g_d = 1.0 + self.gamma * (d_norm ** 2)
        p_keep = torch.pow(bar_beta_t, g_d)
        return torch.clamp(p_keep, min=self.beta_min)


# ============================================================================
# ============================================================================
#                    P C N   E X T R A C T O R
# ============================================================================
# ============================================================================
class ZOE(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size, padding=0)
        self.norm = nn.GroupNorm(1, embed_dim)
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x_t):
        return self.norm(self.proj(x_t)) * self.gate


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
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
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, F_i, t_emb):
        B, C, H, W = F_i.shape
        tokens = F_i.flatten(2).transpose(1, 2)
        time_token = self.time_token_proj(t_emb).unsqueeze(1)
        seq = torch.cat([time_token, tokens], dim=1)

        seq_norm = self.norm1(seq)
        attn_out, _ = self.self_attn(seq_norm, seq_norm, seq_norm)
        seq = seq + attn_out
        seq = seq + self.ffn(self.norm2(seq))

        tokens_out = seq[:, 1:, :]
        return tokens_out.transpose(1, 2).reshape(B, C, H, W)


class PCNExtractor(nn.Module):
    def __init__(self, model_name='pvt_v2_b4', pretrained=True, embed_dim=768,
                 use_time_token_concat=True):
        super().__init__()
        self.use_time_token_concat = use_time_token_concat
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1),
            nn.SiLU(), nn.BatchNorm2d(3),
        )
        print(f"Loading {model_name} from timm (pretrained={pretrained})...")
        self.pvt = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.feature_channels = self.pvt.feature_info.channels()

        self.zoe = ZOE(in_channels=1, embed_dim=self.feature_channels[0], patch_size=4)

        embed_layer = None
        for module in self.pvt.modules():
            if module.__class__.__name__ in ['OverlapPatchEmbed', 'PatchEmbed']:
                embed_layer = module
                break
        if embed_layer is None:
            raise AttributeError("Không tìm thấy Patch Embedding trong timm")
        embed_layer.register_forward_hook(self._zoe_injection_hook)

        self.time_embed = TimeEmbedding(embed_dim=256)
        self.time_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(256, c), nn.SiLU()) for c in self.feature_channels
        ])

        self.TTC_STAGE_INDICES = [2, 3]
        if self.use_time_token_concat:
            self.time_token_layers = nn.ModuleDict({
                str(i): TimeTokenConcatenation(channels=self.feature_channels[i])
                for i in self.TTC_STAGE_INDICES
            })

        self.hf_proj = nn.Conv2d(self.feature_channels[-1], embed_dim, kernel_size=1)
        self._current_zoe = None

    def _zoe_injection_hook(self, module, inputs, outputs):
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        zoe_emb = self._current_zoe
        C = zoe_emb.shape[1]
        if x.dim() == 4:
            if x.shape[1] == C:
                x = x + zoe_emb
            elif x.shape[-1] == C:
                x = x + zoe_emb.permute(0, 2, 3, 1)
        elif x.dim() == 3:
            x = x + zoe_emb.flatten(2).transpose(1, 2)
        if isinstance(outputs, tuple):
            return (x, *outputs[1:])
        return x

    def forward(self, I, M_v, x_t, t, use_hf=False):
        X_input = self.conv_c(torch.cat([I, M_v], dim=1))
        self._current_zoe = self.zoe(x_t)
        features = self.pvt(X_input)
        t_emb = self.time_embed(t)
        fused_features = []
        for i, F_ in enumerate(features):
            if self.use_time_token_concat and i in self.TTC_STAGE_INDICES:
                fused_features.append(self.time_token_layers[str(i)](F_, t_emb))
            else:
                t_scale = self.time_projs[i](t_emb).unsqueeze(-1).unsqueeze(-1)
                fused_features.append(F_ + t_scale)

        if use_hf:
            hf = self.hf_proj(fused_features[-1])
            B_hf, C_hf, H_hf, W_hf = hf.shape
            return hf.view(B_hf, C_hf, -1).permute(0, 2, 1)
        return fused_features


# ============================================================================
# ============================================================================
#                    D N v 1   +   L O S S E S
# ============================================================================
# ============================================================================
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
        reduced_channels = reduced_channels or max(channels // 4, 16)
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, reduced_channels, kernel_size=1),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True),
            ) for _ in pool_sizes
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(reduced_channels * len(pool_sizes), 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feats = []
        for k, conv in zip(self.pool_sizes, self.scale_convs):
            pooled = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
            feats.append(conv(pooled))
        return self.fuse(torch.cat(feats, dim=1))


class OcclusionAwareContextRefinement(nn.Module):
    def __init__(self, channels, local_dilations=(1, 2), global_dilations=(3, 4)):
        super().__init__()
        branch_ch = max(channels // 2, 32)

        def conv_layer(in_ch, out_ch, dilation):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )

        d_l1, d_l2 = local_dilations
        d_g1, d_g2 = global_dilations

        self.local_branch = nn.Sequential(
            conv_layer(channels, branch_ch, d_l1),
            conv_layer(branch_ch, branch_ch, d_l2),
            ChannelAttention(branch_ch),
        )
        self.global_branch = nn.Sequential(
            conv_layer(channels, branch_ch, d_g1),
            conv_layer(branch_ch, branch_ch, d_g2),
            ChannelAttention(branch_ch),
        )
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_ch, kernel_size=1),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            ChannelAttention(branch_ch),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.pyramid_attn = PyramidAttention(channels, pool_sizes=(3, 5, 7))

    def forward(self, F4):
        B, C, H, W = F4.shape
        local_out = self.local_branch(F4)
        global_out = self.global_branch(F4)
        pool_out = self.pool_branch(F4)
        pool_out = F.interpolate(pool_out, size=(H, W), mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([local_out, global_out, pool_out], dim=1))
        attn = self.pyramid_attn(fused)
        return F4 + fused * attn


class LocalEmphasis(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])

    def forward(self, features):
        F1_size = features[0].shape[2:]
        f_up = []
        for i, f in enumerate(features):
            p = self.projs[i](f)
            if i > 0:
                p = F.interpolate(p, size=F1_size, mode='bilinear', align_corners=False)
            f_up.append(p)
        return f_up


class AdaptiveFeatureGate(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, A2, F_up1):
        concat_feat = torch.cat([A2, F_up1], dim=1)
        gate = torch.sigmoid(self.conv2(F.relu(self.conv1(concat_feat))))
        return gate * A2 + (1.0 - gate) * F_up1


class DenoisingNetwork(nn.Module):
    """DN v1 — lightweight U-Net, PCN condition at bottleneck."""
    def __init__(self, pcn_channels=[64, 128, 320, 512], fuse_channels=256, use_ocr=True):
        super().__init__()
        self.use_ocr = use_ocr
        if self.use_ocr:
            self.ocr_module = OcclusionAwareContextRefinement(pcn_channels[-1])
        self.le_module = LocalEmphasis(pcn_channels, fuse_channels)
        self.conv_A3 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1), nn.ReLU())
        self.conv_A2 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1), nn.ReLU())
        self.afg_module = AdaptiveFeatureGate(fuse_channels)

        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bottleneck = nn.Conv2d(256 + fuse_channels, 256, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)

        self.time_embed = TimeEmbedding(embed_dim=256)
        unet_channels = {"enc1": 64, "enc2": 128, "enc3": 256, "bottleneck": 256,
                         "dec1": 128, "dec2": 64}
        self.time_projs = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(256, c), nn.SiLU())
            for name, c in unet_channels.items()
        })

    def forward(self, x_t, t, pcn_features):
        F1, F2, F3, F4 = pcn_features
        if self.use_ocr:
            F4 = self.ocr_module(F4)
        F_up1, F_up2, F_up3, F_up4 = self.le_module([F1, F2, F3, F4])
        A3 = self.conv_A3(torch.cat([F_up4, F_up3], dim=1))
        A2 = self.conv_A2(torch.cat([A3, F_up2], dim=1))
        A1 = self.afg_module(A2, F_up1)

        t_emb = self.time_embed(t)

        def inject_t(feat, name):
            return feat + self.time_projs[name](t_emb).unsqueeze(-1).unsqueeze(-1)

        e1 = F.relu(inject_t(self.enc1(x_t), "enc1"))
        e2 = F.relu(inject_t(self.enc2(e1), "enc2"))
        e3 = F.relu(inject_t(self.enc3(e2), "enc3"))
        b = F.relu(inject_t(self.bottleneck(torch.cat([e3, A1], dim=1)), "bottleneck"))
        d1 = F.relu(inject_t(self.dec1(b), "dec1"))
        d2 = F.relu(inject_t(self.dec2(d1), "dec2"))
        return self.final_conv(d2)


class WeightedBCELoss(nn.Module):
    def forward(self, pred_logits, target_mask, weight_map=None):
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask, reduction='none')
        if weight_map is not None:
            loss = loss * weight_map
        return loss.mean()


class WeightedIoULoss(nn.Module):
    def forward(self, pred_logits, target_mask, weight_map=None):
        pred_probs = torch.sigmoid(pred_logits)
        if weight_map is None:
            weight_map = torch.ones_like(target_mask)
        intersection = (pred_probs * target_mask * weight_map).sum(dim=(2, 3))
        union = ((pred_probs + target_mask) * weight_map).sum(dim=(2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return (1.0 - iou).mean()


class BoundaryAwareEdgeLoss(nn.Module):
    def __init__(self, weight_boundary: float = 2.0):
        super().__init__()
        self.weight_boundary = weight_boundary
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))

    def _extract_edges(self, mask):
        return F.conv2d(mask, self.laplacian, padding=1).abs().clamp(0, 1)

    def forward(self, pred_logits, target_mask, modal_mask):
        pred_edges = self._extract_edges(torch.sigmoid(pred_logits))
        target_edges = self._extract_edges(target_mask)
        modal_edges = self._extract_edges(modal_mask)
        weight_map = 1.0 + (self.weight_boundary - 1.0) * modal_edges
        edge_diff = F.smooth_l1_loss(pred_edges, target_edges, reduction='none')
        return (edge_diff * weight_map).mean()


# ============================================================================
# ============================================================================
#                    D N v 2   ( U - N E T )
# ============================================================================
# ============================================================================
class UNetDenoisingNetwork(nn.Module):
    """DN v2 — U-Net với skip connections + M_v input."""
    def __init__(self, pcn_channels=[64, 128, 320, 512], fuse_channels=256, use_ocr=True):
        super().__init__()
        self.use_ocr = use_ocr
        if self.use_ocr:
            self.ocr_module = OcclusionAwareContextRefinement(pcn_channels[-1])
        self.le_module = LocalEmphasis(pcn_channels, fuse_channels)
        self.conv_A3 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.conv_A2 = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.afg_module = AdaptiveFeatureGate(fuse_channels)

        self.enc1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bottleneck = nn.Conv2d(256 + fuse_channels, 256, kernel_size=3, padding=1)

        self.dec1_up = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec1_fuse = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)
        self.dec2_up = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2_fuse = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.time_embed = TimeEmbedding(embed_dim=256)
        unet_channels = {"enc1": 64, "enc2": 128, "enc3": 256, "bottleneck": 256,
                         "dec1": 128, "dec2": 64}
        self.time_projs = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(256, c), nn.SiLU())
            for name, c in unet_channels.items()
        })

    def forward(self, x_t, t, pcn_features, M_v):
        F1, F2, F3, F4 = pcn_features
        if self.use_ocr:
            F4 = self.ocr_module(F4)
        F_up1, F_up2, F_up3, F_up4 = self.le_module([F1, F2, F3, F4])
        A3 = self.conv_A3(torch.cat([F_up4, F_up3], dim=1))
        A2 = self.conv_A2(torch.cat([A3, F_up2], dim=1))
        A1 = self.afg_module(A2, F_up1)

        t_emb = self.time_embed(t)

        def inject_t(feat, name):
            return feat + self.time_projs[name](t_emb).unsqueeze(-1).unsqueeze(-1)

        unet_input = torch.cat([x_t, M_v], dim=1)
        e1 = F.relu(inject_t(self.enc1(unet_input), "enc1"))
        e2 = F.relu(inject_t(self.enc2(e1), "enc2"))
        e3 = F.relu(inject_t(self.enc3(e2), "enc3"))
        b = F.relu(inject_t(self.bottleneck(torch.cat([e3, A1], dim=1)), "bottleneck"))
        d1_up = F.relu(inject_t(self.dec1_up(b), "dec1"))
        d1 = F.relu(self.dec1_fuse(torch.cat([d1_up, e2], dim=1)))
        d2_up = F.relu(inject_t(self.dec2_up(d1), "dec2"))
        d2 = F.relu(self.dec2_fuse(torch.cat([d2_up, e1], dim=1)))
        return self.final_conv(d2)


def build_denoising_network(variant, pcn_channels, fuse_channels=256, use_ocr=True):
    if variant == "old":
        return DenoisingNetwork(pcn_channels=pcn_channels, fuse_channels=fuse_channels,
                                use_ocr=use_ocr)
    elif variant == "new":
        return UNetDenoisingNetwork(pcn_channels=pcn_channels, fuse_channels=fuse_channels,
                                    use_ocr=use_ocr)
    raise ValueError(f"Unknown variant: {variant}. Use 'old' or 'new'.")


# ============================================================================
# ============================================================================
#                    M E T R I C S
# ============================================================================
# ============================================================================
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


def compute_amodal_metrics(pred_mask, gt_amodal, gt_modal, threshold=0.5):
    pred = (pred_mask > threshold).to(torch.int64)
    gt_a = gt_amodal.to(torch.int64)
    gt_m = gt_modal.to(torch.int64)

    intersection = ((pred == 1) & (gt_a == 1)).sum(dim=(1, 2, 3))
    union = ((pred == 1) | (gt_a == 1)).sum(dim=(1, 2, 3))
    iou_batch = intersection.float() / (union.float() + 1e-6)

    inv_intersection = ((pred == 1) & (gt_a == 1) & (gt_m == 0)).sum(dim=(1, 2, 3))
    inv_union = (((pred == 1) | (gt_a == 1)) & (gt_m == 0)).sum(dim=(1, 2, 3))
    inv_iou_batch = inv_intersection.float() / (inv_union.float() + 1e-6)

    return iou_batch, inv_iou_batch, (inv_union > 0)


# ============================================================================
# ============================================================================
#                    T R A I N I N G   U T I L S
# ============================================================================
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_loss_plot_individual(history, save_path, strategy_name, phase_name):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_steps'], history['train_losses'], alpha=0.6, color='blue')
    plt.xlabel('Steps'); plt.ylabel('Loss')
    plt.title(f'[{strategy_name.upper()} | {phase_name.upper()}] Train Loss (Steps)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 3, 2)
    if len(history['epoch_train_losses']) > 0:
        epochs = range(1, len(history['epoch_train_losses']) + 1)
        plt.plot(epochs, history['epoch_train_losses'], label='Train', marker='o')
        plt.plot(epochs, history['epoch_val_losses'], label='Val', marker='s')
        plt.xlabel('Epochs'); plt.title('Train vs Val Loss')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 3, 3)
    if len(history.get('epoch_miou_inv', [])) > 0:
        epochs = range(1, len(history['epoch_miou_inv']) + 1)
        plt.plot(epochs, history['epoch_miou_full'], label='mIoU_full', marker='o')
        plt.plot(epochs, history['epoch_miou_inv'], label='mIoU_inv', marker='s')
        plt.xlabel('Epochs'); plt.ylabel('mIoU'); plt.ylim(0, 1)
        plt.title('mIoU (fixed t = T-1)')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


class ResumableSampler(Sampler):
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch, start_index=0):
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(indices[self.start_index:])

    def __len__(self):
        return len(self.data_source) - self.start_index


# ============================================================================
# ============================================================================
#                    B U I L D   S Y S T E M
# ============================================================================
# ============================================================================
def build_system(cfg, device, log_fn=print):
    log_fn("=" * 60)
    log_fn(f"🚀 KHỞI TẠO: [{cfg['STRATEGY_NAME'].upper()}] — MODE [{cfg['EXPERIMENT_MODE'].upper()}] "
           f"— DN [{cfg['DN_VARIANT'].upper()}]")
    log_fn("=" * 60)

    if cfg["EXPERIMENT_MODE"] == "zero-shot":
        data_dir_target = cfg["PIX2GESTALT_DIR"]
        mode_target = "pix2gestalt"
    elif cfg["EXPERIMENT_MODE"] == "supervised":
        data_dir_target = cfg["COCOA_TRAIN_JSON"]
        mode_target = "cocoa_train"
    else:
        raise ValueError(f"EXPERIMENT_MODE không hợp lệ: {cfg['EXPERIMENT_MODE']}")

    # ---- Dataset với per-type augmentation flags ----
    train_dataset = AmodalDataset(
        data_dir=data_dir_target,
        mode=mode_target,
        image_size=cfg["IMAGE_SIZE"],
        coco_img_dir=cfg["COCO_IMG_DIR"],
        augment=False,                                       # master tắt để per-type hoạt động độc lập
        augment_hflip=cfg["AUGMENT_HFLIP"],
        augment_scale_crop=cfg["AUGMENT_SCALE_CROP"],
        augment_rotate=cfg["AUGMENT_ROTATE"],
        augment_color_jitter=cfg["AUGMENT_COLOR_JITTER"],
        augment_synthetic_occlusion=cfg["AUGMENT_SYNTHETIC_OCCLUSION"],
        filter_degenerate=cfg["FILTER_DEGENERATE"],
        filter_fully_occluded=cfg["FILTER_FULLY_OCCLUDED"],
        min_area_px=cfg["MIN_AREA_PX"],
    )
    log_fn(f"🔧 Augmentation: hflip={cfg['AUGMENT_HFLIP']} "
           f"scale_crop={cfg['AUGMENT_SCALE_CROP']} "
           f"rotate={cfg['AUGMENT_ROTATE']} "
           f"color_jitter={cfg['AUGMENT_COLOR_JITTER']} "
           f"synthetic_occlusion={cfg['AUGMENT_SYNTHETIC_OCCLUSION']}")

    sampler = ResumableSampler(train_dataset, seed=cfg["SEED"])
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg["BATCH_SIZE"], sampler=sampler,
        drop_last=True, num_workers=cfg["NUM_WORKERS"], pin_memory=True,
        persistent_workers=(cfg["NUM_WORKERS"] > 0),
    )

    val_dataset = AmodalDataset(
        data_dir=cfg["COCOA_VAL_JSON"], mode="cocoa_val",
        image_size=cfg["IMAGE_SIZE"], coco_img_dir=cfg["COCO_IMG_DIR"],
        augment=False,
        filter_degenerate=cfg["FILTER_DEGENERATE"],
        filter_fully_occluded=cfg["FILTER_FULLY_OCCLUDED"],
        min_area_px=cfg["MIN_AREA_PX"],
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False,
        drop_last=False, num_workers=cfg["NUM_WORKERS"], pin_memory=True,
    )

    # ---- Scheduler ----
    if cfg["STRATEGY_NAME"] == "baseline":
        scheduler = BaselineUSTScheduler(num_train_timesteps=cfg["TIMESTEPS"])
    elif cfg["STRATEGY_NAME"] == "clamped":
        scheduler = ClampedDAUSTScheduler(
            num_train_timesteps=cfg["TIMESTEPS"], sigma=5.0,
            beta_min=cfg["DA_UST_BETA_MIN"],
        )
    elif cfg["STRATEGY_NAME"] == "exponential":
        scheduler = ExponentialDAUSTScheduler(
            num_train_timesteps=cfg["TIMESTEPS"],
            gamma=cfg["DA_UST_GAMMA"], beta_min=cfg["DA_UST_BETA_MIN"],
        )
    else:
        raise ValueError(f"STRATEGY_NAME không hợp lệ: {cfg['STRATEGY_NAME']}")

    scheduler.gaussian_betas = torch.linspace(
        cfg["GAUSS_BETA_START"], cfg["GAUSS_BETA_END"], cfg["TIMESTEPS"])
    scheduler.gaussian_alphas_cumprod = torch.cumprod(1 - scheduler.gaussian_betas, dim=0)
    log_fn(f"🔧 Gaussian schedule override: "
           f"alpha_bar[0]={scheduler.gaussian_alphas_cumprod[0]:.6f}  "
           f"alpha_bar[T-1]={scheduler.gaussian_alphas_cumprod[-1]:.6f}  "
           f"sqrt(ab_T-1)={scheduler.gaussian_alphas_cumprod[-1].sqrt():.4f}")

    # ---- Models ----
    pcn = PCNExtractor(model_name='pvt_v2_b4', pretrained=True).to(device)
    dn = build_denoising_network(
        variant=cfg["DN_VARIANT"],
        pcn_channels=pcn.feature_channels,
        fuse_channels=256,
        use_ocr=True,
    ).to(device)

    bce_criterion = WeightedBCELoss().to(device)
    iou_criterion = WeightedIoULoss().to(device)
    edge_criterion = BoundaryAwareEdgeLoss(weight_boundary=2.0).to(device)

    optimizer = optim.AdamW([
        {'params': pcn.parameters(), 'lr': cfg["LEARNING_RATE"] * 0.1},
        {'params': dn.parameters(), 'lr': cfg["LEARNING_RATE"]},
    ], weight_decay=5e-2)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["NUM_EPOCHS"])

    return (train_dataloader, val_dataloader, sampler, scheduler,
            pcn, dn, bce_criterion, iou_criterion, edge_criterion,
            optimizer, lr_scheduler)


# ============================================================================
# ============================================================================
#                    V A L I D A T I O N
# ============================================================================
# ============================================================================
def validate(val_dataloader, scheduler, pcn, dn,
             bce_criterion, iou_criterion, edge_criterion, cfg, device):
    """Deterministic fixed-corruption validation (fork_rng optional)."""
    pcn.eval(); dn.eval()
    val_loss = 0.0
    total_iou_sum, total_iou_count = 0.0, 0
    total_inv_sum, total_inv_count = 0.0, 0

    def _run_body():
        nonlocal val_loss, total_iou_sum, total_iou_count, total_inv_sum, total_inv_count
        with torch.no_grad():
            for batch in val_dataloader:
                I, M_v, M_a = [x.to(device) for x in batch]
                B = I.shape[0]
                t = torch.full((B,), cfg["TIMESTEPS"] - 1, device=device).long()
                distance_map = get_distance_map(M_v)
                x_t = scheduler.add_noise(M_a, distance_map, t)
                weight_map = compute_occlusion_weight_map(
                    M_a, M_v, occlusion_weight=cfg["OCCLUSION_LOSS_WEIGHT"])

                pyramid_features = pcn(I, M_v, x_t, t, use_hf=False)
                if cfg["DN_VARIANT"] == "old":
                    x_hat_0_logits = dn(x_t, t, pyramid_features)
                else:
                    x_hat_0_logits = dn(x_t, t, pyramid_features, M_v)

                loss_bce = bce_criterion(x_hat_0_logits, M_a, weight_map)
                loss_iou = iou_criterion(x_hat_0_logits, M_a, weight_map)
                if cfg["EDGE_LOSS_WEIGHT"] > 0:
                    boundary_ref = M_a if cfg["EDGE_LOSS_BOUNDARY_SOURCE"] == "amodal" else M_v
                    loss_edge = edge_criterion(x_hat_0_logits, M_a, boundary_ref)
                else:
                    loss_edge = torch.tensor(0.0, device=device)
                val_loss += (loss_bce + loss_iou + cfg["EDGE_LOSS_WEIGHT"] * loss_edge).item()

                pred_prob = torch.sigmoid(x_hat_0_logits)
                iou_batch, inv_iou_batch, valid_inv = compute_amodal_metrics(
                    pred_prob, M_a, M_v, threshold=0.5)
                total_iou_sum += iou_batch.sum().item()
                total_iou_count += iou_batch.numel()
                if valid_inv.any():
                    total_inv_sum += inv_iou_batch[valid_inv].sum().item()
                    total_inv_count += int(valid_inv.sum().item())

    if cfg["FORK_RNG_VALIDATION"]:
        if device.type == "cuda":
            cur_idx = device.index if device.index is not None else torch.cuda.current_device()
            fork_devices = [cur_idx]
        else:
            fork_devices = []

        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(cfg["VAL_SEED"])
            if device.type == "cuda":
                torch.cuda.manual_seed_all(cfg["VAL_SEED"])
            _run_body()
    else:
        _run_body()

    return (
        val_loss / len(val_dataloader),
        total_iou_sum / max(total_iou_count, 1),
        total_inv_sum / max(total_inv_count, 1),
    )


# ============================================================================
# ============================================================================
#                    T R A I N   L O O P
# ============================================================================
# ============================================================================
def restore_external_checkpoint(resume_path, latest_ckpt_path, log_fn=print):
    if resume_path is None:
        return
    if not os.path.exists(resume_path):
        log_fn(f"⚠️ Không tìm thấy RESUME_FROM_EXTERNAL: {resume_path}")
        return
    if os.path.exists(latest_ckpt_path):
        log_fn("⚡ Đã có checkpoint trong SAVE_DIR — bỏ qua copy external.")
        return
    shutil.copy(resume_path, latest_ckpt_path)
    log_fn(f"✅ Đã copy checkpoint external: {latest_ckpt_path}")


def train_model(cfg, device, save_dir, resume_path, log_fn=print):
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = save_dir / "latest_ckpt.pth"
    best_ckpt = save_dir / "best_ckpt.pth"
    best_miou_ckpt = save_dir / "best_miou_ckpt.pth"
    history_json = save_dir / "history.json"
    plot_path = save_dir / "loss_chart.png"

    (train_dataloader, val_dataloader, sampler, scheduler,
     pcn, dn, bce_criterion, iou_criterion, edge_criterion,
     optimizer, lr_scheduler) = build_system(cfg, device, log_fn=log_fn)

    start_epoch = 0
    resume_sample_idx = 0
    best_val_loss = float('inf')
    best_miou_inv = 0.0
    epochs_no_improve = 0
    global_step = 0

    history = {
        'train_steps': [], 'train_losses': [],
        'epoch_train_losses': [], 'epoch_val_losses': [],
        'epoch_miou_full': [], 'epoch_miou_inv': [],
    }

    restore_external_checkpoint(resume_path, str(latest_ckpt), log_fn=log_fn)

    if latest_ckpt.exists():
        log_fn(f"♻️ Khôi phục checkpoint [{cfg['STRATEGY_NAME']} | {cfg['EXPERIMENT_MODE']}]...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        pcn.load_state_dict(checkpoint['pcn_state_dict'])
        dn.load_state_dict(checkpoint['dn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        resume_sample_idx = checkpoint.get('resume_sample_idx', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_miou_inv = checkpoint.get('best_miou_inv', 0.0)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        history = checkpoint.get('history', history)
        global_step = checkpoint.get('global_step', 0)
        for k in ('epoch_miou_full', 'epoch_miou_inv'):
            history.setdefault(k, [])
        log_fn(f"✅ Khôi phục: epoch {start_epoch + 1}, "
               f"sample offset {resume_sample_idx}, global_step {global_step}")
    else:
        log_fn("🆕 Không có checkpoint — bắt đầu từ đầu.")

    if start_epoch >= cfg["NUM_EPOCHS"]:
        log_fn(f"🎉 Đã hoàn thành đủ {cfg['NUM_EPOCHS']} epochs.")
        return history

    global_start_time = time.time()
    time_limit_seconds = cfg["MAX_TRAIN_HOURS"] * 3600
    grad_clip = cfg["GRAD_CLIP_NORM"]

    for epoch in range(start_epoch, cfg["NUM_EPOCHS"]):
        if (time.time() - global_start_time) > time_limit_seconds:
            log_fn(f"\n⏰ Hết giờ! Tạm ngắt ở epoch {epoch + 1}.")
            return history

        offset = resume_sample_idx if epoch == start_epoch else 0
        sampler.set_epoch(epoch, start_index=offset)

        pcn.train(); dn.train()
        train_loss = 0.0
        pbar = tqdm(train_dataloader,
                    desc=f"[{cfg['STRATEGY_NAME'].upper()}|{cfg['EXPERIMENT_MODE'].upper()}] "
                         f"Ep {epoch+1}/{cfg['NUM_EPOCHS']}")
        time_up = False

        for batch_idx, batch in enumerate(pbar):
            I, M_v, M_a = [x.to(device) for x in batch]
            optimizer.zero_grad()
            B = I.shape[0]

            u = torch.rand(B, device=device)
            t = (cfg["TIMESTEPS"] * (u ** 0.5)).long().clamp(0, cfg["TIMESTEPS"] - 1)
            distance_map = get_distance_map(M_v)
            weight_map = compute_occlusion_weight_map(
                M_a, M_v, occlusion_weight=cfg["OCCLUSION_LOSS_WEIGHT"])

            x_t = scheduler.add_noise(M_a, distance_map, t)
            pyramid_features = pcn(I, M_v, x_t, t, use_hf=False)
            if cfg["DN_VARIANT"] == "old":
                x_hat_0_logits = dn(x_t, t, pyramid_features)
            else:
                x_hat_0_logits = dn(x_t, t, pyramid_features, M_v)

            loss_bce = bce_criterion(x_hat_0_logits, M_a, weight_map)
            loss_iou = iou_criterion(x_hat_0_logits, M_a, weight_map)
            if cfg["EDGE_LOSS_WEIGHT"] > 0:
                boundary_ref = M_a if cfg["EDGE_LOSS_BOUNDARY_SOURCE"] == "amodal" else M_v
                loss_edge = edge_criterion(x_hat_0_logits, M_a, boundary_ref)
            else:
                loss_edge = torch.tensor(0.0, device=device)
            loss = loss_bce + loss_iou + cfg["EDGE_LOSS_WEIGHT"] * loss_edge

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(pcn.parameters()) + list(dn.parameters()), grad_clip)
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                history['train_steps'].append(global_step)
                history['train_losses'].append(loss.item())

            time_up = (time.time() - global_start_time) > time_limit_seconds - 300

            if global_step % 500 == 0 or time_up:
                resume_sample_idx = offset + (batch_idx + 1) * cfg["BATCH_SIZE"]
                torch.save({
                    'mode': cfg["EXPERIMENT_MODE"],
                    'strategy': cfg["STRATEGY_NAME"],
                    'dn_variant': cfg["DN_VARIANT"],
                    'epoch': epoch,
                    'resume_sample_idx': resume_sample_idx,
                    'global_step': global_step,
                    'pcn_state_dict': pcn.state_dict(),
                    'dn_state_dict': dn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_miou_inv': best_miou_inv,
                    'epochs_no_improve': epochs_no_improve,
                    'history': history,
                }, str(latest_ckpt))

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if time_up:
                log_fn(f"\n⏰ Hết giờ giữa epoch — lưu checkpoint sample {resume_sample_idx}.")
                return history

        # End of epoch
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss, miou_full, miou_inv = validate(
            val_dataloader, scheduler, pcn, dn,
            bce_criterion, iou_criterion, edge_criterion, cfg, device)
        lr_scheduler.step()

        history['epoch_train_losses'].append(avg_train_loss)
        history['epoch_val_losses'].append(avg_val_loss)
        history['epoch_miou_full'].append(miou_full)
        history['epoch_miou_inv'].append(miou_inv)

        log_fn(f"📊 Epoch {epoch + 1}: Train = {avg_train_loss:.5f} | "
               f"Val = {avg_val_loss:.5f} | "
               f"mIoU_full = {miou_full:.4f} | mIoU_inv = {miou_inv:.4f} "
               f"[F:seed={cfg['VAL_SEED']}]")

        save_loss_plot_individual(history, str(plot_path),
                                   cfg["STRATEGY_NAME"], cfg["EXPERIMENT_MODE"])

        improved_loss = avg_val_loss < best_val_loss
        if improved_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        improved_miou = miou_inv > best_miou_inv
        if improved_miou:
            best_miou_inv = miou_inv

        checkpoint_data = {
            'mode': cfg["EXPERIMENT_MODE"],
            'strategy': cfg["STRATEGY_NAME"],
            'dn_variant': cfg["DN_VARIANT"],
            'epoch': epoch + 1,
            'resume_sample_idx': 0,
            'global_step': global_step,
            'pcn_state_dict': pcn.state_dict(),
            'dn_state_dict': dn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_miou_inv': best_miou_inv,
            'epochs_no_improve': epochs_no_improve,
            'history': history,
        }
        torch.save(checkpoint_data, str(latest_ckpt))

        if improved_loss:
            torch.save(checkpoint_data, str(best_ckpt))
            log_fn(f"🏆 Best loss checkpoint: {best_val_loss:.5f}")

        if improved_miou:
            torch.save(checkpoint_data, str(best_miou_ckpt))
            log_fn(f"🏆 Best mIoU_inv checkpoint: {best_miou_inv:.4f}")

        with open(history_json, "w") as f:
            json.dump(history, f, indent=2)

        if cfg["EARLY_STOP_PATIENCE"] > 0 and epochs_no_improve >= cfg["EARLY_STOP_PATIENCE"]:
            log_fn(f"\n🛑 EARLY STOPPING tại epoch {epoch + 1} "
                   f"(best_val_loss={best_val_loss:.5f}, best_miou_inv={best_miou_inv:.4f}).")
            break

    log_fn(f"\n🎉 Hoàn tất [{cfg['STRATEGY_NAME']} | {cfg['EXPERIMENT_MODE']}].")
    del pcn, dn, optimizer, scheduler, train_dataloader, val_dataloader
    torch.cuda.empty_cache(); gc.collect()
    return history


# ============================================================================
# ============================================================================
#                    C L I   +   E N T R Y
# ============================================================================
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="CondDiff-AMO training (single-file, self-contained).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=str, default=None,
                   help="Override DATA_ROOT (parent của coco_amodal/ và pix2gestalt/)")
    p.add_argument("--save-dir", type=str, default=None, help="Override SAVE_DIR")
    p.add_argument("--resume", type=str, default=None, help="Override RESUME_FROM_EXTERNAL")
    p.add_argument("--max-hours", type=float, default=None, help="Override MAX_TRAIN_HOURS")
    p.add_argument("--epochs", type=int, default=None, help="Override NUM_EPOCHS")
    p.add_argument("--batch-size", type=int, default=None, help="Override BATCH_SIZE")
    return p.parse_args()


def _build_cfg_from_globals():
    """Gom các biến global ở section CONFIG thành dict."""
    g = globals()
    keys = [
        "EXPERIMENT_MODE", "STRATEGY_NAME", "DN_VARIANT", "SEED",
        "NUM_EPOCHS", "LEARNING_RATE", "BATCH_SIZE", "IMAGE_SIZE", "TIMESTEPS",
        "MAX_TRAIN_HOURS", "NUM_WORKERS",
        "OCCLUSION_LOSS_WEIGHT", "EDGE_LOSS_WEIGHT", "EDGE_LOSS_BOUNDARY_SOURCE",
        "EARLY_STOP_PATIENCE",
        "DA_UST_GAMMA", "DA_UST_BETA_MIN",
        "GAUSS_BETA_START", "GAUSS_BETA_END",
        "VAL_SEED", "FORK_RNG_VALIDATION", "GRAD_CLIP_NORM",
        "AUGMENT_HFLIP", "AUGMENT_SCALE_CROP", "AUGMENT_ROTATE",
        "AUGMENT_COLOR_JITTER", "AUGMENT_SYNTHETIC_OCCLUSION",
        "FILTER_DEGENERATE", "FILTER_FULLY_OCCLUDED", "MIN_AREA_PX",
        "COCO_IMG_DIR", "PIX2GESTALT_DIR", "COCOA_TRAIN_JSON", "COCOA_VAL_JSON",
        "SAVE_DIR", "RESUME_FROM_EXTERNAL",
    ]
    return {k: g[k] for k in keys}


def main():
    args = parse_args()

    # ---- Apply CLI overrides vào globals ----
    g = globals()
    if args.save_dir is not None:
        g["SAVE_DIR"] = args.save_dir
    if args.resume is not None:
        g["RESUME_FROM_EXTERNAL"] = args.resume
    if args.max_hours is not None:
        g["MAX_TRAIN_HOURS"] = args.max_hours
    if args.epochs is not None:
        g["NUM_EPOCHS"] = args.epochs
    if args.batch_size is not None:
        g["BATCH_SIZE"] = args.batch_size
    if args.data_root is not None:
        data_root = _Path(args.data_root).resolve()
        g["DATA_ROOT"] = data_root
        g["COCO_IMG_DIR"] = str(data_root / "coco_amodal" / "cocoa_images_extracted")
        g["PIX2GESTALT_DIR"] = str(data_root / "pix2gestalt" / "pix2gestalt_occlusions_release")
        g["COCOA_TRAIN_JSON"] = str(data_root / "coco_amodal" / "COCO_amodal_train2014.json")
        g["COCOA_VAL_JSON"] = str(data_root / "coco_amodal" / "COCO_amodal_val2014.json")

    cfg = _build_cfg_from_globals()

    set_seed(cfg["SEED"])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    save_dir = _Path(cfg["SAVE_DIR"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎯 CONFIG")
    print("=" * 60)
    print(f"  Mode              : {cfg['EXPERIMENT_MODE']}")
    print(f"  Strategy          : {cfg['STRATEGY_NAME']} "
          f"(gamma={cfg['DA_UST_GAMMA']}, beta_min={cfg['DA_UST_BETA_MIN']})")
    print(f"  DN variant        : {cfg['DN_VARIANT']}")
    print(f"  Gaussian schedule : ({cfg['GAUSS_BETA_START']}, {cfg['GAUSS_BETA_END']})")
    print(f"  Epochs            : {cfg['NUM_EPOCHS']}")
    print(f"  Batch / LR        : {cfg['BATCH_SIZE']} / {cfg['LEARNING_RATE']}")
    print(f"  Max hours         : {cfg['MAX_TRAIN_HOURS']}")
    print(f"  Occlusion weight  : {cfg['OCCLUSION_LOSS_WEIGHT']}")
    print(f"  Grad clip         : {cfg['GRAD_CLIP_NORM']}")
    print(f"  Fork RNG val      : {cfg['FORK_RNG_VALIDATION']}")
    print(f"  Device            : {device}")
    print(f"  Save dir          : {save_dir}")
    print(f"  Data dir (COCOA)  : {cfg['COCO_IMG_DIR']}")
    print(f"  Resume from       : {cfg['RESUME_FROM_EXTERNAL']}")
    print("=" * 60)

    train_model(cfg, device, save_dir, cfg["RESUME_FROM_EXTERNAL"], log_fn=print)

    print("\n🏁 KẾT THÚC.")
    print(f"   latest_ckpt.pth    : {save_dir / 'latest_ckpt.pth'}")
    print(f"   best_ckpt.pth      : {save_dir / 'best_ckpt.pth'}")
    print(f"   best_miou_ckpt.pth : {save_dir / 'best_miou_ckpt.pth'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Dừng bởi người dùng.")
        sys.exit(130)
    except Exception:
        print("\n[FATAL] train_all_in_one.py failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)