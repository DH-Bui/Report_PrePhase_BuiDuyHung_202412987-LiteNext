class LabeledDataset(data.Dataset):
    """Dataset for labeled data with weak augmentation"""
    def __init__(self, images, masks, weak_transform=None):
        self.images = images
        self.masks = masks
        self.weak_transform = weak_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].astype(np.uint8)
        mask = self.masks[index].squeeze().astype(np.float32)

        # Apply weak augmentation
        aug_weak = self.weak_transform(image=image, mask=mask)
        image_weak = aug_weak["image"]
        mask_weak = aug_weak["mask"].unsqueeze(0).float()

        return image_weak, mask_weak

class UnlabeledDataset(data.Dataset):
    """Dataset for unlabeled data with strong and weak augmentation"""
    def __init__(self, images, strong_transform=None, weak_transform=None):
        self.images = images
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].astype(np.uint8)

        # Apply strong augmentation
        image_strong = self.strong_transform(image=image)["image"]

        # Apply weak augmentation
        image_weak = self.weak_transform(image=image)["image"]

        return image_strong, image_weak

class TestDataset(data.Dataset):
    """Dataset for test data"""
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].astype(np.uint8)
        mask = self.masks[index].squeeze().astype(np.float32)

        # Apply transform
        aug = self.transform(image=image, mask=mask)
        image_test = aug["image"]
        mask_test = aug["mask"].unsqueeze(0).float()

        return image_test, mask_test
