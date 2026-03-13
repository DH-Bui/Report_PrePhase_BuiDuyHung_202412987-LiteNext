import numpy as np
import torch
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import multiprocessing
import matplotlib.pyplot as plt

def get_dataloaders(batch_size=2, img_size=256):
    """
    Create dataloaders that return exactly the components requested.

    Returns:
        unlabeled_dataloader: Returns (image_unlabel_strong, image_unlabel_weak)
        labeled_dataloader: Returns (image_label_weak, mask_label_weak)
        test_dataloader: Returns (image_test, mask_test)
    """
    # Load data
    x_train = np.load('/content/drive/MyDrive/Project_IPSAL_1/data science bowl 2018/train_1.npy')
    y_train1 = np.load('/content/drive/MyDrive/Project_IPSAL_1/data science bowl 2018/train_mask_1.npy')
    x_test = np.load('/content/drive/MyDrive/Project_IPSAL_1/data science bowl 2018/test_real.npy')
    y_test1 = np.load('/content/drive/MyDrive/Project_IPSAL_1/data science bowl 2018/test_real_mask.npy')

    # Process masks if needed
    if y_train1.ndim == 4 and y_train1.shape[3] > 1:
        y_train = y_train1[:,:,:,0]
    else:
        y_train = y_train1

    if y_test1.ndim == 4 and y_test1.shape[3] > 1:
        y_test = y_test1[:,:,:,0]
    else:
        y_test = y_test1

    # Split training data into labeled (50%) and unlabeled (50%)
    x_labeled, x_unlabeled, y_labeled, _ = train_test_split(
        x_train, y_train, test_size=0.8, random_state=42
    )

    # Print dataset statistics
    print("Dataset statistics:")
    print(f"  Total training samples: {len(x_train)}")
    print(f"  Labeled samples: {len(x_labeled)}")
    print(f"  Unlabeled samples: {len(x_unlabeled)}")
    print(f"  Test samples: {len(x_test)}")

    # Define transformations
    # Strong augmentation
    strong_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(p=0.2),
        A.Normalize(
            mean=(0., 0., 0.),
            std=(1., 1., 1.),
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    # Weak augmentation
    weak_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.3),
        A.Normalize(
            mean=(0., 0., 0.),
            std=(1., 1., 1.),
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    # Test transform
    test_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=(0., 0., 0.),
            std=(1., 1., 1.),
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    # Create datasets
    labeled_dataset = LabeledDataset(
        images=x_labeled,
        masks=y_labeled,
        weak_transform=weak_transform
    )

    unlabeled_dataset = UnlabeledDataset(
        images=x_unlabeled,
        strong_transform=strong_transform,
        weak_transform=weak_transform
    )

    test_dataset = TestDataset(
        images=x_test,
        masks=y_test,
        transform=test_transform
    )

    # Create dataloaders
    labeled_dataloader = data.DataLoader(
        dataset=labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=multiprocessing.Pool()._processes,
    )

    unlabeled_dataloader = data.DataLoader(
        dataset=unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=multiprocessing.Pool()._processes,
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=multiprocessing.Pool()._processes,
    )

    return unlabeled_dataloader, labeled_dataloader, test_dataloader
