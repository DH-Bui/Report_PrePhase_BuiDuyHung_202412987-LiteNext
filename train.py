from data.dataloader import get_dataloaders

class Args:
    def __init__(self):
        self.root = "/content/drive/MyDrive/Pytorch_ML_BDH/Report_PrePhase_BuiDuyHung/Dataset/"
        self.data_paths = {
            'BUSI':   self.root + "BUSI",
            'ISIC':   self.root + "ISIC",
            'REFUGE': self.root + "REFUGE",
            'DSB':    "/content/drive/MyDrive/Project_IPSAL_1/data science bowl 2018"
        }
        
        self.dataset = "BUSI" 
        self.epochs = 300
        self.batch_size = 8
        self.epochs = 300
        self.batch_size = 8    # Batch size cho tập Labeled
        self.mu = 4            # Tỉ lệ Unlabeled/Labeled (Ví dụ batch_u = 8 * 4 = 32)
        self.img_size = 256
        
        # --- Optimizer & Scheduler ---
        self.lr = 4e-4
        self.lrs_min = 1e-7
        self.type_lr = "LROnP"
        self.optim = "NAdam"
        
        # --- Kiến trúc Model ---
        self.encoder_block = "LiteNeXt"
        self.mgpu = False
        
        # --- BYOL ---
        self.labeled_ratio = 0.2 # Chỉ dùng 20% nhãn để học
        self.ema_decay = 0.999   # Hệ số cập nhật Teacher

    def get_path(self):
        return self.data_paths.get(self.dataset)

args = Args(dataset="BUSI")

u_loader, l_loader, t_loader = get_dataloaders(args)
