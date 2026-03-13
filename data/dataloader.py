import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .dataset import LabeledDataset, UnlabeledDataset, TestDataset
from .augmentations import get_transforms

def get_dataloaders(args):
    base_path = args.get_path()
    if base_path is None:
        raise ValueError(f"Dataset {args.dataset} không có trong danh sách mapping!")

    x_train_all = np.load(os.path.join(base_path, "x_train.npy"))
    y_train_all = np.load(os.path.join(base_path, "y_train.npy"))
    x_test = np.load(os.path.join(base_path, "x_test.npy"))
    y_test = np.load(os.path.join(base_path, "y_test.npy"))

    u_ratio = 1 - args.labeled_ratio
    x_labeled, x_unlabeled, y_labeled, _ = train_test_split(
        x_train_all, y_train_all, 
        test_size=u_ratio, 
        random_state=42
    )

    strong_tf, weak_tf, test_tf = get_transforms(args.img_size)

    labeled_set = LabeledDataset(x_labeled, y_labeled, weak_tf)
    unlabeled_set = UnlabeledDataset(x_unlabeled, strong_tf, weak_tf)
    test_set = TestDataset(x_test, y_test, test_tf)

    l_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    u_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    t_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return u_loader, l_loader, t_loader
