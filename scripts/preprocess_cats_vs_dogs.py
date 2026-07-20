import os
import shutil
import yaml
import random
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# cats_vs_dogs 数据集预处理脚本
# 1) 读取配置文件
# 2) 检查原始图像是否损坏
# 3) 收集有效图像并按类别汇总
# 4) 随机打乱后按比例划分 train/val/test
# 5) 将图像统一调整大小并保存到处理后目录
# 6) 生成 labels.csv, images_info.csv, dataset_stats.yaml 等元数据文件

def load_config():
    """加载 cats_vs_dogs 数据集配置文件"""
    config_path = "config/datasets/cats_vs_dogs.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def check_corrupted_images(image_path):
    """检查图像是否损坏"""
    try:
        img = Image.open(image_path)
        img.verify()  # 验证图像完整性
        return True
    except:
        return False

def preprocess_cats_vs_dogs():
    """预处理 cats_vs_dogs 数据集"""
    
    # 加载配置
    config = load_config()
    
    # 创建处理后的目录结构
    processed_dir = Path(config['processed_dir'])
    for subdir in ['train', 'val', 'test', 'images']:
        (processed_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 原始数据路径
    raw_data_dir = Path(config['raw_data_dir'])
    pet_images_dir = raw_data_dir / "PetImages"
    
    # 检查是否存在损坏的图片并收集有效数据
    print("检查图像文件并收集有效数据...")
    valid_images = []
    class_id_map = {"cat": 0, "dog": 1}
    
    for class_name, class_id in class_id_map.items():
        class_dir = pet_images_dir / class_name.capitalize()
        
        if not class_dir.exists():
            class_dir = pet_images_dir / class_name  # 尝试小写目录名
            if not class_dir.exists():
                print(f"警告: {class_name} 目录不存在于 {class_dir}")
                continue
        
        print(f"处理 {class_name} 类别...")
        image_files = list(class_dir.glob("*.jpg"))
        print(f"  找到 {len(image_files)} 个图像文件")
        
        valid_count = 0
        for img_path in image_files:
            if check_corrupted_images(img_path):
                valid_images.append({
                    'path': img_path,
                    'class_name': class_name,
                    'class_id': class_id
                })
                valid_count += 1
            else:
                print(f"  损坏的图像: {img_path.name}")
        
        print(f"  有效图像: {valid_count}/{len(image_files)}")
    
    print(f"总共有效图像: {len(valid_images)}")
    
    if len(valid_images) == 0:
        raise ValueError("没有找到有效的图像文件！")
    
    # 随机打乱数据
    random.shuffle(valid_images)
    
    # 划分训练集、验证集、测试集
    train_ratio = config['train_split']
    val_ratio = config['val_split']
    test_ratio = config['test_split']
    
    # 计算划分数量
    n_total = len(valid_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = valid_images[:n_train]
    val_data = valid_images[n_train:n_train + n_val]
    test_data = valid_images[n_train + n_val:]
    
    print(f"\n数据划分结果:")
    print(f"  训练集: {len(train_data)} 张图片")
    print(f"  验证集: {len(val_data)} 张图片")
    print(f"  测试集: {len(test_data)} 张图片")
    
    # 定义内部函数：将每个子集的图像按需处理并保存到指定目录。
    # 处理逻辑包括读取图像、转换为 RGB、统一尺寸、保存文件，并构建元数据信息。
    def process_and_save_images(image_list, split_name):
        """处理并保存图像到指定目录"""
        split_dir = processed_dir / split_name
        image_info = []
        
        for idx, img_info in enumerate(image_list):
            try:
                # 读取图像
                img_path = img_info['path']
                img = cv2.imread(str(img_path))
                
                if img is None:
                    print(f"  无法读取: {img_path}")
                    continue
                
                # 转换为RGB（如果需要）
                if len(img.shape) == 2:  # 灰度图
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:  # BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 调整大小
                target_size = tuple(config['image_size'])
                img_resized = cv2.resize(img, target_size)
                
                # 生成新文件名
                class_name = img_info['class_name']
                new_filename = f"{class_name}_{split_name}_{idx:05d}.jpg"
                save_path = split_dir / new_filename
                
                # 保存图像
                cv2.imwrite(str(save_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                
                # 记录信息
                image_info.append({
                    'filename': new_filename,
                    'original_path': str(img_path),
                    'class_name': class_name,
                    'class_id': img_info['class_id'],
                    'split': split_name,
                    'height': img_resized.shape[0],
                    'width': img_resized.shape[1]
                })
                
            except Exception as e:
                print(f"  处理 {img_path} 时出错: {e}")
                continue
        
        return image_info
    
    print("\n处理并保存图像...")
    
    # 处理各个数据集
    train_info = process_and_save_images(train_data, 'train')
    val_info = process_and_save_images(val_data, 'val')
    test_info = process_and_save_images(test_data, 'test')
    
    # 合并所有信息
    all_info = train_info + val_info + test_info
    
    # 创建标签映射文件
    labels_df = pd.DataFrame({
        'class_id': [0, 1],
        'class_name': ['cat', 'dog']
    })
    labels_path = processed_dir / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    
    # 创建图像信息文件
    info_df = pd.DataFrame(all_info)
    info_path = processed_dir / "images_info.csv"
    info_df.to_csv(info_path, index=False)
    
    # 创建数据集统计文件
    stats = {
        'total_images': len(all_info),
        'train_images': len(train_info),
        'val_images': len(val_info),
        'test_images': len(test_info),
        'classes': config['num_classes'],
        'class_names': config['class_names'],
        'image_size': config['image_size']
    }
    
    stats_path = processed_dir / "dataset_stats.yaml"
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    print("\n" + "="*50)
    print("预处理完成！")
    print(f"处理后的数据保存在: {processed_dir}")
    print(f"标签文件: {labels_path}")
    print(f"图像信息文件: {info_path}")
    print(f"数据集统计: {stats_path}")
    print("="*50)
    
    # 显示类别的分布
    print("\n类别分布:")
    for split_name, split_info in [('train', train_info), ('val', val_info), ('test', test_info)]:
        if split_info:
            split_df = pd.DataFrame(split_info)
            print(f"\n{split_name}:")
            class_counts = split_df['class_name'].value_counts().to_dict()
            for class_name in ['cat', 'dog']:
                count = class_counts.get(class_name, 0)
                percentage = count / len(split_info) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    preprocess_cats_vs_dogs()