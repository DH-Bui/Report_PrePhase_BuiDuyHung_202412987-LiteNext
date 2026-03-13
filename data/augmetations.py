import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size=256):
    # 1. Bộ cực mạnh cho Student (Unlabeled)
    strong_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # 2. Bộ nhẹ cho Teacher (Unlabeled) và Labeled Train
    weak_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # 3. Bộ sạch cho Test/Validation (CHƯA CÓ TRONG CODE CỦA ÔNG)
    test_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Trả về đủ 3 món để không bị lỗi "unpack"
    return strong_transform, weak_transform, test_transform
