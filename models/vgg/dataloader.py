"""
宠物图像数据集加载器
支持三个预处理后的数据集: cats_vs_dogs, oxford_pets, stanford_dogs
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import pandas as pd

class PetDataset(Dataset):
    """宠物数据集类"""
    
    def __init__(self, dataset_name='cats_vs_dogs', split='train', data_root='data/processed', augment=False):
        """
        初始化数据集
        
        Args:
            dataset_name: 数据集名称 ('cats_vs_dogs', 'oxford_pets', 'stanford_dogs','doges77')
            split: 数据划分 ('train', 'val', 'test')
            data_root: 数据根目录
            augment: 是否应用数据增强（仅对训练集有效）
        """
        super().__init__()
        
        # 设置路径
        self.data_dir = Path(data_root) / dataset_name
        self.split = split
        self.augment = augment and split == 'train'
        
        # 加载图像信息文件
        info_file = self.data_dir / 'images_info.csv'
        if not info_file.exists():
            raise FileNotFoundError(f"找不到图像信息文件: {info_file}")
        
        info_df = pd.read_csv(info_file)
        
        # 筛选当前split的数据
        self.info_df = info_df[info_df['split'] == split].reset_index(drop=True)
        
        if len(self.info_df) == 0:
            raise ValueError(f"在{split}划分中没有找到数据")
        
        # 获取图像路径和标签
        self.image_paths = []
        self.labels = []
        
        for _, row in self.info_df.iterrows():
            img_path = self.data_dir / split / row['filename']
            if img_path.exists():
                self.image_paths.append(img_path)
                self.labels.append(int(row['class_id']))
        
        # 加载标签映射
        labels_file = self.data_dir / 'labels.csv'
        self.labels_df = pd.read_csv(labels_file)
        self.num_classes = len(self.labels_df)
        
        print(f"[{dataset_name}] {split}集: {len(self.image_paths)}张图片, {self.num_classes}个类别")
        
        # 设置数据变换
        self.transform = self._get_transform()
    
    def _get_transform(self):
        """获取数据变换"""
        # 基础变换：调整大小 + 转Tensor + 归一化
        base_transform = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
        
        if self.augment:
            # === 数据增强 START (仅训练集有效) ===
            # 训练集：在此处添加/修改数据增强策略，方便以后扩展
            transform_list = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1)
            ),
        ] + base_transform
            # === 数据增强 END ===
        else:
            # 验证/测试集：只有基础变换
            transform_list = base_transform
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            # 返回空白图像作为占位符
            return torch.zeros(3, 224, 224), label
    
    def get_class_counts(self):
        """获取每个类别的样本数量"""
        from collections import Counter
        return Counter(self.labels)


def get_dataloaders(dataset_name='cats_vs_dogs', batch_size=32, augment=False):
    """
    获取训练、验证、测试数据加载器
    
    Args:
        dataset_name: 数据集名称
        batch_size: 批量大小
        augment: 是否对训练集使用数据增强
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    data_root = 'data/processed'
    
    # 创建数据集
    train_dataset = PetDataset(dataset_name, 'train', data_root, augment=augment)
    val_dataset = PetDataset(dataset_name, 'val', data_root, augment=False)
    test_dataset = PetDataset(dataset_name, 'test', data_root, augment=False)
    
    num_classes = train_dataset.num_classes
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=False
    )
    
    return train_loader, val_loader, test_loader, num_classes


def show_dataset_info(dataset_name='cats_vs_dogs'):
    """显示数据集基本信息"""
    from collections import Counter
    
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print('='*60)
    
    data_root = 'data/processed'
    
    # 加载标签文件
    labels_file = Path(data_root) / dataset_name / 'labels.csv'
    labels_df = pd.read_csv(labels_file)
    
    print(f"总类别数: {len(labels_df)}")
    
    # 显示前几个类别
    print("\n前5个类别:")
    for _, row in labels_df.head().iterrows():
        if 'breed_name' in row:
            print(f"  {row['class_id']}: {row['breed_name']}")
        elif 'class_name' in row:
            print(f"  {row['class_id']}: {row['class_name']}")
    
    # 统计各划分的样本数量
    splits = ['train', 'val', 'test']
    for split in splits:
        dataset = PetDataset(dataset_name, split, data_root, augment=False)
        class_counts = dataset.get_class_counts()
        print(f"\n{split.upper()}集: {len(dataset)}张图片")
        print(f"类别分布 (样本数前5):")
        for class_id, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  类别{class_id}: {count}张")
    
    print('='*60)


# 使用示例
if __name__ == "__main__":
    # 显示数据集信息
    show_dataset_info('cats_vs_dogs')
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset_name='cats_vs_dogs',
        batch_size=16,
        augment=True
    )
    
    # 测试一个batch
    for images, labels in train_loader:
        print(f"\nBatch形状: {images.shape}")
        print(f"标签: {labels[:10]}...")  # 显示前10个标签
        break
