#!/usr/bin/env python
"""
预处理 Doges 77 Breeds 数据集
处理9个分片的合并、去重、划分
"""

import os
import shutil
import yaml
import random
import hashlib
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Doges 77 Breeds 数据集预处理脚本
# 1) 扫描多个原始分片目录
# 2) 检查损坏图像并计算文件哈希去重
# 3) 按配置划分 train/val/test
# 4) 统一调整图像尺寸并保存处理后数据
# 5) 生成标签映射、图像信息和统计文件

def load_config():
    """加载 Doges 77 Breeds 数据集配置文件"""
    config_path = "config/datasets/doges77.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_file_hash(file_path):
    """计算文件的MD5哈希值用于去重"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # 分块读取，避免大文件占用太多内存
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_corrupted_images(image_path):
    """检查图像是否损坏"""
    try:
        img = Image.open(image_path)
        img.verify()  # 验证图像完整性
        # 验证后需要重新打开，因为verify()会关闭文件
        img = Image.open(image_path)
        img.load()  # 尝试加载像素数据
        return True
    except Exception as e:
        print(f"  损坏的图像 {image_path}: {e}")
        return False

def scan_all_shards(raw_data_dir):
    """
    扫描所有分片，收集图像信息并去重
    返回：图像信息列表，品种列表
    """
    raw_data_dir = Path(raw_data_dir)
    
    # 查找所有分片目录（train-20230705T073235Z-XXX）
    shard_dirs = sorted([d for d in raw_data_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('train-')])
    
    print(f"找到 {len(shard_dirs)} 个分片目录:")
    for d in shard_dirs:
        train_subdir = d / "train"
        if train_subdir.exists():
            breed_count = len(list(train_subdir.iterdir())) if train_subdir.exists() else 0
            print(f"  {d.name}: {breed_count} 个品种目录")
    
    # 收集所有图像信息
    all_images = []  # 包含所有图像的信息（可能有重复）
    breed_names = set()  # 所有品种名称
    
    # 先扫描所有图像，建立文件哈希到路径的映射
    print("\n扫描所有分片中的图像...")
    
    for shard_idx, shard_dir in enumerate(shard_dirs):
        train_dir = shard_dir / "train"
        if not train_dir.exists():
            print(f"警告: {shard_dir.name} 中没有 train 目录")
            continue
        
        print(f"\n处理分片 {shard_idx+1}/{len(shard_dirs)}: {shard_dir.name}")
        
        # 遍历该分片中的所有品种目录
        breed_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        print(f"  找到 {len(breed_dirs)} 个品种目录")
        
        for breed_dir in breed_dirs:
            breed_name = breed_dir.name
            breed_names.add(breed_name)
            
            # 获取该品种的所有jpg图像
            image_files = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.jpeg")) + list(breed_dir.glob("*.png"))
            
            for img_path in image_files:
                all_images.append({
                    'path': img_path,
                    'breed': breed_name,
                    'shard': shard_dir.name,
                    'filename': img_path.name
                })
    
    print(f"\n扫描完成，共找到 {len(all_images)} 个图像文件")
    print(f"共 {len(breed_names)} 个品种")
    
    return all_images, sorted(list(breed_names))

def deduplicate_images(image_list):
    """
    基于MD5哈希去重
    返回：去重后的图像列表，重复统计信息
    """
    print("\n开始去重...")
    
    # 先检查图像是否损坏
    valid_images = []
    corrupted_count = 0
    
    print("检查图像损坏情况...")
    for i, img_info in enumerate(image_list):
        if i % 1000 == 0:
            print(f"  已检查 {i}/{len(image_list)} 张图像")
        
        if check_corrupted_images(img_info['path']):
            valid_images.append(img_info)
        else:
            corrupted_count += 1
    
    print(f"损坏图像: {corrupted_count} 张")
    print(f"有效图像: {len(valid_images)} 张")
    
    # 计算哈希去重
    hash_dict = defaultdict(list)
    hash_duplicates = defaultdict(list)
    
    print("计算文件哈希值...")
    for i, img_info in enumerate(valid_images):
        if i % 1000 == 0:
            print(f"  已处理 {i}/{len(valid_images)} 张图像")
        
        try:
            file_hash = get_file_hash(img_info['path'])
            hash_dict[file_hash].append(img_info)
        except Exception as e:
            print(f"  计算哈希失败: {img_info['path']} - {e}")
            # 如果哈希计算失败，仍保留该图像（使用路径作为唯一标识）
            hash_dict[str(img_info['path'])].append(img_info)
    
    # 找出重复的
    unique_images = []
    duplicate_count = 0
    duplicate_stats = defaultdict(int)
    
    for file_hash, images in hash_dict.items():
        if len(images) == 1:
            unique_images.append(images[0])
        else:
            duplicate_count += len(images) - 1
            # 保留第一个（通常是从较早分片来的）
            unique_images.append(images[0])
            
            # 记录重复的品种分布（用于统计）
            for img in images[1:]:
                duplicate_stats[img['breed']] += 1
            
            # 打印重复信息（前几个作为示例）
            if duplicate_count <= 10:
                print(f"  发现重复: {images[0]['breed']} - {images[0]['filename']} "
                      f"(共 {len(images)} 个副本)")
    
    print(f"\n去重完成:")
    print(f"  去重前: {len(valid_images)} 张")
    print(f"  去重后: {len(unique_images)} 张")
    print(f"  移除重复: {duplicate_count} 张")
    
    # 打印重复最多的几个品种
    if duplicate_stats:
        print("\n重复最多的品种:")
        top_duplicates = sorted(duplicate_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        for breed, count in top_duplicates:
            print(f"  {breed}: {count} 张重复")
    
    return unique_images, duplicate_count, corrupted_count

def split_dataset(image_list, train_ratio, val_ratio, random_seed=42):
    """
    划分数据集为 train/val/test。
    先随机打乱图像列表，再根据比例分割，剩余部分作为测试集。
    """
    random.seed(random_seed)
    random.shuffle(image_list)
    
    n_total = len(image_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = image_list[:n_train]
    val_data = image_list[n_train:n_train + n_val]
    test_data = image_list[n_train + n_val:]
    
    return train_data, val_data, test_data

def get_breed_to_class_id(breed_names):
    """
    创建品种名到类别ID的映射（从0开始）。
    该映射用于训练标签并生成 labels.csv。
    """
    breed_to_id = {breed: idx for idx, breed in enumerate(sorted(breed_names))}
    return breed_to_id

def process_and_save_images(image_list, split_name, processed_dir, target_size, breed_to_id):
    """
    处理并保存图像到指定目录
    逻辑：读取图像 -> 转换为 RGB -> 调整大小 -> 生成安全文件名 -> 保存 -> 记录元数据信息
    """
    split_dir = processed_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    saved_images = []
    failed_count = 0
    
    print(f"\n处理 {split_name} 集 ({len(image_list)} 张图像)...")
    
    for idx, img_info in enumerate(image_list):
        if idx % 1000 == 0:
            print(f"  已处理 {idx}/{len(image_list)}")
        
        try:
            img_path = img_info['path']
            breed_name = img_info['breed']
            class_id = breed_to_id[breed_name]
            
            # 读取图像
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"  无法读取: {img_path}")
                failed_count += 1
                continue
            
            # 转换为RGB
            if len(img.shape) == 2:  # 灰度图
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:  # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整大小
            target_size = tuple(target_size)
            img_resized = cv2.resize(img, target_size)
            
            # 生成新文件名
            # 格式: {breed_name}_{class_id:03d}_{split}_{idx:06d}.jpg
            safe_breed_name = breed_name.replace(' ', '_').replace('/', '_')
            new_filename = f"{safe_breed_name}_{class_id:03d}_{split_name}_{idx:06d}.jpg"
            save_path = split_dir / new_filename
            
            # 保存图像
            cv2.imwrite(str(save_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            
            # 记录信息
            saved_images.append({
                'filename': new_filename,
                'original_path': str(img_path),
                'original_shard': img_info.get('shard', 'unknown'),
                'breed': breed_name,
                'class_id': class_id,
                'split': split_name,
                'height': img_resized.shape[0],
                'width': img_resized.shape[1]
            })
            
        except Exception as e:
            print(f"  处理 {img_path} 时出错: {e}")
            failed_count += 1
            continue
    
    print(f"  {split_name} 集完成: 成功 {len(saved_images)} / 失败 {failed_count}")
    return saved_images, failed_count

def preprocess_doges77():
    """主预处理函数"""
    
    # 加载配置
    config = load_config()
    processed_dir = Path(config['processed_dir'])
    raw_data_dir = Path(config['raw_data_dir'])
    
    # 创建处理后的目录结构
    for subdir in ['train', 'val', 'test', 'images']:
        (processed_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("开始预处理 Doges 77 Breeds 数据集")
    print("="*60)
    
    # 步骤1: 扫描所有分片
    all_images, breed_names = scan_all_shards(raw_data_dir)
    
    if len(all_images) == 0:
        print("错误: 没有找到任何图像文件！")
        return
    
    # 步骤2: 去重
    unique_images, duplicate_count, corrupted_count = deduplicate_images(all_images)
    
    # 步骤3: 创建品种到类别ID的映射
    breed_to_id = get_breed_to_class_id(breed_names)
    print(f"\n品种映射: {len(breed_to_id)} 个类别")
    
    # 打印前10个品种及其ID
    print("\n前10个品种:")
    for breed, idx in list(breed_to_id.items())[:10]:
        print(f"  {idx:3d}: {breed}")
    
    # 步骤4: 划分数据集
    train_ratio = config['train_split']
    val_ratio = config['val_split']
    test_ratio = config['test_split']
    
    train_data, val_data, test_data = split_dataset(
        unique_images, train_ratio, val_ratio, config['random_seed']
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_data)} 张 ({train_ratio*100:.0f}%)")
    print(f"  验证集: {len(val_data)} 张 ({val_ratio*100:.0f}%)")
    print(f"  测试集: {len(test_data)} 张 ({test_ratio*100:.0f}%)")
    
    # 步骤5: 处理并保存图像
    target_size = config['image_size']
    
    train_saved, train_failed = process_and_save_images(
        train_data, 'train', processed_dir, target_size, breed_to_id
    )
    val_saved, val_failed = process_and_save_images(
        val_data, 'val', processed_dir, target_size, breed_to_id
    )
    test_saved, test_failed = process_and_save_images(
        test_data, 'test', processed_dir, target_size, breed_to_id
    )
    
    # 合并所有保存的图像信息
    all_saved = train_saved + val_saved + test_saved
    
    # 步骤6: 创建标签映射文件
    labels_data = []
    for breed, class_id in breed_to_id.items():
        labels_data.append({
            'class_id': class_id,
            'breed_name': breed
        })
    
    labels_df = pd.DataFrame(labels_data)
    labels_path = processed_dir / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"\n标签文件已保存: {labels_path}")
    
    # 步骤7: 创建图像信息文件
    info_df = pd.DataFrame(all_saved)
    info_path = processed_dir / "images_info.csv"
    info_df.to_csv(info_path, index=False)
    print(f"图像信息文件已保存: {info_path}")
    
    # 步骤8: 创建数据集统计文件
    # 统计每个品种的数量
    breed_counts = Counter([img['breed'] for img in all_saved])
    
    stats = {
        'dataset_name': config['name'],
        'total_images': len(all_saved),
        'train_images': len(train_saved),
        'val_images': len(val_saved),
        'test_images': len(test_saved),
        'num_classes': len(breed_to_id),
        'image_size': target_size,
        'original_images_found': len(all_images),
        'corrupted_images': corrupted_count,
        'duplicates_removed': duplicate_count,
        'train_failed': train_failed,
        'val_failed': val_failed,
        'test_failed': test_failed,
        'breed_distribution': {
            breed: {
                'count': count,
                'class_id': breed_to_id[breed]
            }
            for breed, count in breed_counts.items()
        }
    }
    
    stats_path = processed_dir / "dataset_stats.yaml"
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)
    print(f"统计文件已保存: {stats_path}")
    
    # 步骤9: 打印最终统计
    print("\n" + "="*60)
    print("预处理完成！")
    print("="*60)
    
    print(f"\n最终数据集统计:")
    print(f"  总图像数: {len(all_saved)}")
    print(f"  类别数: {len(breed_to_id)}")
    print(f"  训练集: {len(train_saved)} 张")
    print(f"  验证集: {len(val_saved)} 张")
    print(f"  测试集: {len(test_saved)} 张")
    
    # 显示每个品种的样本量（前10个）
    print("\n各品种样本量（前10个）:")
    top_breeds = breed_counts.most_common(10)
    for breed, count in top_breeds:
        class_id = breed_to_id[breed]
        print(f"  {class_id:3d} - {breed:30s}: {count:4d} 张")
    
    # 显示样本量最少的几个品种
    print("\n样本量最少的品种（后5个）:")
    bottom_breeds = breed_counts.most_common()[-5:]
    for breed, count in bottom_breeds:
        class_id = breed_to_id[breed]
        print(f"  {class_id:3d} - {breed:30s}: {count:4d} 张")
    
    print(f"\n处理后的数据保存在: {processed_dir}")

if __name__ == "__main__":
    preprocess_doges77()