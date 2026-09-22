# CondDiff-AMO

Mã nguồn phục vụ huấn luyện và thực nghiệm mô hình **Conditional Diffusion cho Amodal Segmentation**, sử dụng các thiết lập UST/DA-UST và mạng Denoising Network.

Repository hỗ trợ hai chế độ thực nghiệm chính:

- **Supervised:** huấn luyện trên COCOA.
- **Zero-shot:** huấn luyện trên Pix2Gestalt.

---

## 1. Yêu cầu môi trường

Khuyến nghị:

- Python 3.10 hoặc 3.11
- GPU NVIDIA
- CUDA tương thích với phiên bản PyTorch trên máy
- Linux / Ubuntu đối với môi trường server

Các thư viện Python cần thiết được liệt kê trong:

```text
requirements.txt
```

---

# 2. Cài đặt

## 2.1. Clone repository

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

## 2.2. Tạo virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Nếu sử dụng Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 2.3. Cài đặt PyTorch

PyTorch nên được cài đặt phù hợp với phiên bản CUDA của máy/server.

Ví dụ:

```bash
pip install torch torchvision
```

Kiểm tra PyTorch và GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Nếu máy có GPU và PyTorch nhận được GPU, kết quả sẽ hiển thị:

```text
CUDA: True
```

> **Lưu ý:** Trên server/cluster có CUDA module riêng, nên sử dụng phiên bản PyTorch phù hợp với môi trường CUDA được cung cấp bởi server.

## 2.4. Cài đặt các thư viện còn lại

```bash
pip install -r requirements.txt
```

---

# 3. Cấu trúc thư mục

Sau khi chuẩn bị đầy đủ dataset, cấu trúc thư mục được kỳ vọng như sau:

```text
project/
├── train_all_in_one.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── coco_amodal/
│   │   ├── COCO_amodal_train2014.json
│   │   ├── COCO_amodal_val2014.json
│   │   └── cocoa_images_extracted/
│   │       ├── ...
│   │       └── ...
│   │
│   └── pix2gestalt/
│       └── pix2gestalt_occlusions_release/
│           ├── occlusion/
│           ├── visible_object_mask/
│           └── whole_mask/
│
└── checkpoints/
```

Thư mục `checkpoints/` sẽ được chương trình tạo trong quá trình huấn luyện nếu chưa tồn tại.

---

# 4. Chuẩn bị COCOA

COCOA được sử dụng cho:

- Supervised training.
- Validation.

Trong project này, chỉ những ảnh được tham chiếu bởi các file annotation cần thiết mới được sử dụng.

Hai file JSON có kích thước nhỏ và được lưu trực tiếp trong repository:

```text
COCO_amodal_train2014.json
COCO_amodal_val2014.json
```

Phần ảnh COCOA được lưu riêng trên Google Drive.

---

## 4.1. Tải ảnh COCOA

Ảnh COCOA đã được lọc được cung cấp tại Google Drive:

**COCOA filtered images**

https://drive.google.com/file/d/1cE5KS4blKI1hJ6zTnjjrARaXtgiHaJcc/view?usp=sharing

Trước tiên cài `gdown`:

```bash
pip install gdown
```

Tạo thư mục chứa COCOA:

```bash
mkdir -p data/coco_amodal
cd data/coco_amodal
```

Tải file:

```bash
gdown "https://drive.google.com/file/d/1cE5KS4blKI1hJ6zTnjjrARaXtgiHaJcc/view?usp=sharing"
```

Sau khi tải xong:

```bash
ls -lh
```

---

## 4.2. Giải nén COCOA

```bash
unzip COCOA_IMAGES.zip
```

Sau khi giải nén, cần đảm bảo thư mục ảnh có dạng:

```text
data/
└── coco_amodal/
    ├── COCO_amodal_train2014.json
    ├── COCO_amodal_val2014.json
    └── cocoa_images_extracted/
        ├── ...
        └── ...
```

> Tên file archive có thể thay đổi tùy theo file được lưu trên Google Drive. Quan trọng là thư mục ảnh cuối cùng phải có tên `cocoa_images_extracted` để khớp với cấu hình mặc định của project.

---

# 5. Chuẩn bị Pix2Gestalt

Pix2Gestalt được sử dụng cho experiment **zero-shot**.

Dataset có thể được tải trực tiếp từ nguồn chính thức:

```text
https://gestalt.cs.columbia.edu/assets/pix2gestalt_occlusions_release.tar.gz
```

Tạo thư mục:

```bash
mkdir -p data/pix2gestalt
cd data/pix2gestalt
```

Tải dataset:

```bash
wget https://gestalt.cs.columbia.edu/assets/pix2gestalt_occlusions_release.tar.gz
```

Giải nén:

```bash
tar -xvf pix2gestalt_occlusions_release.tar.gz
```

Sau khi giải nén, cấu trúc cần có:

```text
data/
└── pix2gestalt/
    └── pix2gestalt_occlusions_release/
        ├── occlusion/
        ├── visible_object_mask/
        └── whole_mask/
```

### Ý nghĩa các thư mục

| Thư mục | Nội dung |
|---|---|
| `occlusion/` | Ảnh đầu vào có đối tượng bị che khuất |
| `visible_object_mask/` | Visible/modal mask |
| `whole_mask/` | Amodal/whole-object mask |

Dataset loader sẽ ghép các thành phần tương ứng dựa trên ID của sample.

---

# 6. Kiểm tra dataset

Từ thư mục gốc của project:

```bash
cd <PROJECT_ROOT>
```

Kiểm tra COCOA:

```bash
ls data/coco_amodal
```

Cần có:

```text
COCO_amodal_train2014.json
COCO_amodal_val2014.json
cocoa_images_extracted
```

Kiểm tra Pix2Gestalt:

```bash
ls data/pix2gestalt/pix2gestalt_occlusions_release
```

Cần có:

```text
occlusion
visible_object_mask
whole_mask
```

Nếu các thư mục trên tồn tại đúng vị trí, cấu trúc dataset cơ bản đã sẵn sàng.

---

# 7. Cấu hình training

Các cấu hình chính được đặt trong file training.

Một số thiết lập quan trọng:

```python
NUM_EPOCHS      = 40
LEARNING_RATE   = 1e-4
BATCH_SIZE      = 16
IMAGE_SIZE      = (256, 256)
TIMESTEPS       = 1000
MAX_TRAIN_HOURS = 10.0
```

Strategy:

```python
STRATEGY_NAME = "exponential"
```

Denoising Network:

```python
DN_VARIANT = "new"
```

---

# 8. Supervised training với COCOA

Để huấn luyện supervised trên COCOA:

```python
EXPERIMENT_MODE = "supervised"
```

Dataset training:

```text
data/coco_amodal/COCO_amodal_train2014.json
```

Dataset validation:

```text
data/coco_amodal/COCO_amodal_val2014.json
```

Sau khi cấu hình xong, chạy:

```bash
python train_all_in_one.py
```

---

# 9. Zero-shot experiment với Pix2Gestalt

Để sử dụng Pix2Gestalt cho training:

```python
EXPERIMENT_MODE = "zero-shot"
```

Dataset training sẽ được lấy từ:

```text
data/pix2gestalt/pix2gestalt_occlusions_release/
```

Validation vẫn sử dụng COCOA validation:

```text
data/coco_amodal/COCO_amodal_val2014.json
```

Chạy:

```bash
python train_all_in_one.py
```

---

# 10. Thay đổi vị trí dataset

Nếu toàn bộ dataset được đặt ở một thư mục khác, có thể truyền `--data-root`:

```bash
python train_all_in_one.py --data-root /path/to/data
```

Khi đó chương trình kỳ vọng cấu trúc:

```text
/path/to/data/
├── coco_amodal/
│   ├── COCO_amodal_train2014.json
│   ├── COCO_amodal_val2014.json
│   └── cocoa_images_extracted/
│
└── pix2gestalt/
    └── pix2gestalt_occlusions_release/
        ├── occlusion/
        ├── visible_object_mask/
        └── whole_mask/
```

---

# 11. Smoke test

Trước khi chạy experiment trong thời gian dài, nên kiểm tra pipeline bằng một epoch:

```bash
python train_all_in_one.py --epochs 1
```

Nếu GPU có ít VRAM hơn dự kiến, có thể giảm batch size:

```bash
python train_all_in_one.py --epochs 1 --batch-size 4
```

Smoke test nên xác nhận rằng:

1. Dataset được đọc thành công.
2. Model được khởi tạo.
3. Denoising Network được khởi tạo.
4. Forward pass hoạt động.
5. Loss được tính thành công.
6. Backward pass hoạt động.
7. Validation hoạt động.
8. Checkpoint được lưu.

Chỉ nên chạy experiment đầy đủ sau khi smoke test hoàn thành mà không có lỗi.

---

# 12. Resume training

Checkpoint được lưu trong:

```text
checkpoints/
```

Có thể tiếp tục training từ checkpoint bằng:

```bash
python train_all_in_one.py --resume /path/to/checkpoint.pth
```

Ví dụ:

```bash
python train_all_in_one.py \
    --resume checkpoints/latest_ckpt.pth
```

---

# 13. Giới hạn thời gian training

Có thể đặt giới hạn thời gian chạy bằng:

```bash
python train_all_in_one.py --max-hours 6
```

Ví dụ trên server có giới hạn 6 giờ:

```bash
python train_all_in_one.py --max-hours 6
```

Checkpoint sẽ được lưu trong quá trình training để có thể tiếp tục experiment ở lần chạy sau.

---

# 14. Augmentation

Cấu hình augmentation hiện tại:

```python
AUGMENT_HFLIP               = True
AUGMENT_SCALE_CROP          = False
AUGMENT_ROTATE              = False
AUGMENT_COLOR_JITTER        = False
AUGMENT_SYNTHETIC_OCCLUSION = False
```

Do đó hiện tại chỉ sử dụng **horizontal flip**.

Các augmentation khác được tắt mặc định để giữ pipeline thực nghiệm ổn định.

---

# 15. Dataset filtering

Các tùy chọn filtering hiện tại:

```python
FILTER_DEGENERATE     = False
FILTER_FULLY_OCCLUDED = False
MIN_AREA_PX           = 4
```

Mặc định không bật:

- Loại sample suy biến.
- Loại sample bị che khuất hoàn toàn.

Không nên thay đổi các thiết lập này nếu mục tiêu là tái lập experiment với cấu hình hiện tại.

---

# 16. Kết quả và checkpoint

Các kết quả training được lưu trong thư mục:

```text
checkpoints/
```

Tùy theo quá trình training, thư mục có thể chứa:

```text
checkpoints/
├── latest_ckpt.pth
├── best_ckpt.pth
├── best_miou_ckpt.pth
├── history.json
└── loss_chart.png
```

---

# 17. Quy trình nhanh

Nếu dataset đã được chuẩn bị, quy trình chạy project có thể rút gọn thành:

### Bước 1 — Clone repository

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

### Bước 2 — Cài dependencies

```bash
pip install -r requirements.txt
```

### Bước 3 — Chuẩn bị COCOA

Tải ảnh từ Google Drive:

```bash
gdown "https://drive.google.com/file/d/1cE5KS4blKI1hJ6zTnjjrARaXtgiHaJcc/view?usp=sharing"
```

Giải nén vào:

```text
data/coco_amodal/cocoa_images_extracted/
```

### Bước 4 — Chuẩn bị Pix2Gestalt

```bash
cd data/pix2gestalt

wget https://gestalt.cs.columbia.edu/assets/pix2gestalt_occlusions_release.tar.gz

tar -xvf pix2gestalt_occlusions_release.tar.gz
```

### Bước 5 — Smoke test

```bash
python train_all_in_one.py --epochs 1
```

### Bước 6 — Training

Supervised:

```python
EXPERIMENT_MODE = "supervised"
```

hoặc zero-shot:

```python
EXPERIMENT_MODE = "zero-shot"
```

Sau đó:

```bash
python train_all_in_one.py
```

---

# 18. Checklist

Trước khi bắt đầu training, kiểm tra:

- [ ] Repository đã được clone.
- [ ] Python environment đã được tạo.
- [ ] PyTorch nhận GPU.
- [ ] `requirements.txt` đã được cài đặt.
- [ ] `COCO_amodal_train2014.json` tồn tại.
- [ ] `COCO_amodal_val2014.json` tồn tại.
- [ ] `cocoa_images_extracted/` tồn tại.
- [ ] Pix2Gestalt đã được tải nếu chạy zero-shot.
- [ ] `occlusion/` tồn tại.
- [ ] `visible_object_mask/` tồn tại.
- [ ] `whole_mask/` tồn tại.
- [ ] Smoke test chạy thành công.

Sau khi hoàn thành checklist, có thể bắt đầu experiment chính.