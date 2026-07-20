import os
import shutil
import yaml
import random
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict

# Oxford Pets 数据集预处理脚本
# 1) 从 annotations/list.txt 中读取图像与类别信息
# 2) 使用 trainval/test 划分文件构建数据集划分
# 3) 进一步将 trainval 划分为 train 和 val
# 4) 统一调整图像尺寸并保存处理后的图像
# 5) 生成 labels.csv, images_info.csv, dataset_stats.yaml 等元数据文件

def load_config():
    """加载 Oxford Pets 数据集配置文件"""
    config_path = "config/datasets/oxford_pets.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def parse_list_file(list_path):
    """解析 list.txt 文件，读取图像名称、类别 ID、物种与品种信息"""
    images_info = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                image_name = parts[0]
                # 原始文件里 class_id 从 1 开始，这里减1使其从0开始
                class_id = int(parts[1]) - 1
                species = int(parts[2])  # 1:Cat, 2:Dog
                breed_id = int(parts[3])
                
                # 从文件名提取类别名（根据README：大写首字母是猫，小写是狗）
                # 实际格式如：Abyssinian_1.jpg
                if '_' in image_name:
                    breed_name = image_name.split('_')[0]
                else:
                    breed_name = image_name.split('.')[0]
                
                images_info.append({
                    'image_name': image_name,
                    'class_id': class_id,
                    'species': species,
                    'breed_id': breed_id,
                    'breed_name': breed_name,
                    'full_path': None  # 稍后填充
                })
    
    return images_info

def parse_split_file(split_path):
    """解析 trainval.txt 或 test.txt 文件，获取官方划分的图像列表"""
    split_images = []
    with open(split_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # 去掉可能的扩展名
                img_name = line.split()[0] if ' ' in line else line
                if not img_name.lower().endswith('.jpg'):
                    img_name += '.jpg'
                split_images.append(img_name)
    return split_images

def check_corrupted_images(image_path):
    """检查图像是否损坏"""
    try:
        img = Image.open(image_path)
        img.verify()  # 验证图像完整性
        return True
    except:
        return False

# 该函数用于创建 class_id -> 品种名称的映射，后续会用于 labels.csv 生成
def get_breed_mapping(images_info):
    """创建品种 ID 到品种名称的映射，用于生成标签文件"""
    breed_map = {}
    for info in images_info:
        breed_id = info['class_id']  # 实际上class_id就是品种ID
        breed_name = info['breed_name']
        breed_map[breed_id] = breed_name
    
    # 排序以确保一致性
    sorted_breeds = sorted(breed_map.items(), key=lambda x: x[0])
    return dict(sorted_breeds)

def preprocess_oxford_pets():
    """预处理 Oxford Pets 数据集"""
    
    # 加载配置
    config = load_config()
    
    # 创建处理后的目录结构
    processed_dir = Path(config['processed_dir'])
    for subdir in ['train', 'val', 'test', 'images']:
        (processed_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 原始数据路径
    raw_data_dir = Path(config['raw_data_dir'])
    images_dir = raw_data_dir / "images"
    annotations_dir = raw_data_dir / "annotations"
    
    # 检查目录是否存在
    if not images_dir.exists():
        print(f"错误: 图像目录不存在 {images_dir}")
        return
    
    if not annotations_dir.exists():
        print(f"错误: 标注目录不存在 {annotations_dir}")
        return
    
    # 解析list.txt文件
    list_file = annotations_dir / "list.txt"
    if not list_file.exists():
        print(f"错误: list.txt 文件不存在 {list_file}")
        return
    
    print("解析list.txt文件...")
    images_info = parse_list_file(list_file)
    print(f"找到 {len(images_info)} 条图像记录")
    
    # 获取品种映射
    breed_map = get_breed_mapping(images_info)
    print(f"共 {len(breed_map)} 个品种")
    
    # 打印前几个品种作为示例
    print("\n前10个品种:")
    for i, (breed_id, breed_name) in enumerate(list(breed_map.items())[:10]):
        print(f"  {breed_id:2d}: {breed_name}")
    
    # 解析划分文件
    trainval_file = annotations_dir / "trainval.txt"
    test_file = annotations_dir / "test.txt"
    
    if not trainval_file.exists() or not test_file.exists():
        print("错误: 缺少划分文件")
        return
    
    trainval_images = parse_split_file(trainval_file)
    test_images = parse_split_file(test_file)
    
    print(f"\n官方划分:")
    print(f"  trainval集: {len(trainval_images)} 张图片")
    print(f"  test集: {len(test_images)} 张图片")
    
    # 检查是否有重叠
    overlap = set(trainval_images) & set(test_images)
    if overlap:
        print(f"  警告: 训练集和测试集有 {len(overlap)} 张图片重叠")
    
    # 验证所有图像文件都存在并收集有效图像
    print("\n检查图像文件...")
    all_images = []
    missing_images = []
    
    for img_info in images_info:
        img_name = img_info['image_name']
        img_path = images_dir / img_name
        
        if not img_path.exists():
            # 尝试添加.jpg扩展名
            if not img_name.lower().endswith('.jpg'):
                img_path = images_dir / (img_name + '.jpg')
            
            if not img_path.exists():
                missing_images.append(img_name)
                continue
        
        # 检查图像是否损坏
        if not check_corrupted_images(img_path):
            print(f"  损坏的图像: {img_name}")
            continue
        
        # 更新完整路径
        img_info['full_path'] = img_path
        
        # 确定划分
        if img_name in trainval_images:
            split = 'trainval'
        elif img_name in test_images:
            split = 'test'
        else:
            # 如果没有在划分文件中，随机分配
            split = 'trainval' if random.random() < 0.8 else 'test'
        
        img_info['split'] = split
        all_images.append(img_info)
    
    print(f"有效图像: {len(all_images)}/{len(images_info)}")
    if missing_images:
        print(f"缺失图像: {len(missing_images)} 张")
        if len(missing_images) <= 10:
            for img in missing_images[:10]:
                print(f"  - {img}")
    
    # 将 trainval 进一步划分为 train 和 val，保持 train/val/test 的最终比例
    # 这里使用 80% 的 trainval 做训练，20% 做验证
    train_images = []
    val_images = []
    test_images_processed = []
    
    for img_info in all_images:
        if img_info['split'] == 'trainval':
            # 80%训练，20%验证（从trainval中划分）
            if random.random() < 0.8:
                img_info['final_split'] = 'train'
                train_images.append(img_info)
            else:
                img_info['final_split'] = 'val'
                val_images.append(img_info)
        else:  # test
            img_info['final_split'] = 'test'
            test_images_processed.append(img_info)
    
    print(f"\n最终划分:")
    print(f"  训练集: {len(train_images)} 张图片")
    print(f"  验证集: {len(val_images)} 张图片")
    print(f"  测试集: {len(test_images_processed)} 张图片")
    
    # 处理并保存图像
    def process_and_save_images(image_list, split_name):
        """处理并保存图像到指定目录"""
        split_dir = processed_dir / split_name
        image_info = []
        
        for idx, img_info in enumerate(image_list):
            try:
                img_path = img_info['full_path']
                if img_path is None:
                    continue
                
                # 读取图像
                img = cv2.imread(str(img_path))
                
                if img is None:
                    print(f"  无法读取: {img_path}")
                    continue
                
                # 转换为RGB
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
                breed_name = img_info['breed_name']
                class_id = img_info['class_id']
                new_filename = f"{breed_name}_{class_id:02d}_{split_name}_{idx:05d}.jpg"
                save_path = split_dir / new_filename
                
                # 保存图像
                cv2.imwrite(str(save_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                
                # 记录信息
                image_info.append({
                    'filename': new_filename,
                    'original_name': img_info['image_name'],
                    'original_path': str(img_path),
                    'breed_name': breed_name,
                    'class_id': class_id,
                    'species': 'cat' if img_info['species'] == 1 else 'dog',
                    'breed_id': img_info['breed_id'],
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
    train_info = process_and_save_images(train_images, 'train')
    val_info = process_and_save_images(val_images, 'val')
    test_info = process_and_save_images(test_images_processed, 'test')
    
    # 合并所有信息
    all_info = train_info + val_info + test_info
    
    # 创建标签映射文件
    labels_data = []
    for class_id, breed_name in breed_map.items():
        # 确定物种（根据README：大写首字母是猫）
        if breed_name[0].isupper():
            species = 'cat'
        else:
            species = 'dog'
        
        labels_data.append({
            'class_id': class_id,
            'breed_name': breed_name,
            'species': species
        })
    
    labels_df = pd.DataFrame(labels_data)
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
        'species_count': {
            'cats': len([i for i in all_info if i['species'] == 'cat']),
            'dogs': len([i for i in all_info if i['species'] == 'dog'])
        },
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
    
    # 显示统计信息
    print("\n数据集统计:")
    print(f"总图像数: {len(all_info)}")
    print(f"猫品种数: {len([i for i in all_info if i['species'] == 'cat'])}")
    print(f"狗品种数: {len([i for i in all_info if i['species'] == 'dog'])}")
    
    # 显示每个split的分布
    print("\n划分分布:")
    for split_name, split_info in [('train', train_info), ('val', val_info), ('test', test_info)]:
        if split_info:
            split_df = pd.DataFrame(split_info)
            print(f"\n{split_name} ({len(split_info)}张):")
            
            # 物种分布
            species_counts = split_df['species'].value_counts()
            for species, count in species_counts.items():
                percentage = count / len(split_info) * 100
                print(f"  {species}: {count} ({percentage:.1f}%)")
    
    # 显示类别分布（前10个）
    print("\n品种分布（前10个）:")
    all_df = pd.DataFrame(all_info)
    breed_counts = all_df['breed_name'].value_counts().head(10)
    for breed, count in breed_counts.items():
        percentage = count / len(all_info) * 100
        print(f"  {breed:20s}: {count:3d} ({percentage:.1f}%)")

if __name__ == "__main__":
    preprocess_oxford_pets()