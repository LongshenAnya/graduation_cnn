import os
import shutil
import yaml
import random
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import scipy.io as sio
from collections import defaultdict

def load_config():
    """加载配置文件"""
    config_path = "config/datasets/stanford_dogs.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def parse_mat_file(mat_path):
    """解析MATLAB文件获取文件列表"""
    try:
        mat_data = sio.loadmat(mat_path)
        
        # 不同的.mat文件可能有不同的结构
        if 'file_list' in mat_data:
            file_list = mat_data['file_list']
        elif 'files' in mat_data:
            file_list = mat_data['files']
        else:
            # 尝试获取第一个数组
            for key in mat_data.keys():
                if not key.startswith('__'):
                    file_list = mat_data[key]
                    break
        
        # 转换MATLAB cell数组为Python列表
        files = []
        for item in file_list:
            # 处理嵌套数组
            if isinstance(item, np.ndarray):
                if item.size > 0:
                    file_name = str(item[0])
                    # 去除可能的填充字符
                    file_name = file_name.strip()
                    files.append(file_name)
        
        return files
    
    except Exception as e:
        print(f"解析MAT文件失败 {mat_path}: {e}")
        return []

def extract_breed_from_path(file_path):
    """从文件路径中提取品种信息"""
    # 示例: n02085620-Chihuahua/n02085620_5927.jpg
    parts = file_path.split('/')
    if len(parts) >= 2:
        breed_folder = parts[0]
        # 格式: n02085620-Chihuahua
        if '-' in breed_folder:
            breed_name = breed_folder.split('-')[1]
            # 提取品种ID（n02085620部分）
            breed_id_part = breed_folder.split('-')[0]
            return breed_name, breed_id_part
    return None, None

def get_all_breeds(images_dir):
    """获取所有品种的文件夹"""
    breeds = []
    breed_id_map = {}
    
    for item in images_dir.iterdir():
        if item.is_dir():
            # 格式: n02085620-Chihuahua
            folder_name = item.name
            if '-' in folder_name:
                breed_id = folder_name.split('-')[0]
                breed_name = folder_name.split('-')[1]
                breeds.append({
                    'folder_name': folder_name,
                    'breed_name': breed_name,
                    'breed_id': breed_id,
                    'folder_path': item
                })
                breed_id_map[breed_id] = breed_name
    
    # 按品种名排序
    breeds.sort(key=lambda x: x['breed_name'])
    
    # 为每个品种分配连续的class_id，从0开始
    for idx, breed in enumerate(breeds):
        breed['class_id'] = idx
    
    return breeds, breed_id_map

def check_corrupted_images(image_path):
    """检查图像是否损坏"""
    try:
        img = Image.open(image_path)
        img.verify()  # 验证图像完整性
        return True
    except:
        return False

def preprocess_stanford_dogs():
    """预处理 Stanford Dogs 数据集"""
    
    # 加载配置
    config = load_config()
    
    # 创建处理后的目录结构
    processed_dir = Path(config['processed_dir'])
    for subdir in ['train', 'val', 'test', 'images']:
        (processed_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 原始数据路径
    raw_data_dir = Path(config['raw_data_dir'])
    images_dir = raw_data_dir / "Images"
    lists_dir = raw_data_dir / "lists"
    
    # 检查目录是否存在
    if not images_dir.exists():
        print(f"错误: 图像目录不存在 {images_dir}")
        return
    
    if not lists_dir.exists():
        print(f"错误: 列表目录不存在 {lists_dir}")
        return
    
    # 获取所有品种
    print("扫描品种文件夹...")
    breeds, breed_id_map = get_all_breeds(images_dir)
    print(f"找到 {len(breeds)} 个狗品种")
    
    # 打印前几个品种作为示例
    print("\n前10个品种:")
    for breed in breeds[:10]:
        print(f"  {breed['class_id']:3d}: {breed['breed_name']} ({breed['breed_id']})")
    
    # 解析划分文件
    train_list_path = lists_dir / "train_list.mat"
    test_list_path = lists_dir / "test_list.mat"
    
    if not train_list_path.exists() or not test_list_path.exists():
        print("错误: 缺少划分文件")
        return
    
    print("\n解析划分文件...")
    train_files = parse_mat_file(train_list_path)
    test_files = parse_mat_file(test_list_path)
    
    print(f"训练集文件数: {len(train_files)}")
    print(f"测试集文件数: {len(test_files)}")
    
    # 检查是否有重叠
    overlap = set(train_files) & set(test_files)
    if overlap:
        print(f"警告: 训练集和测试集有 {len(overlap)} 个文件重叠")
    
    # 创建文件到划分的映射
    file_split_map = {}
    for file in train_files:
        file_split_map[file] = 'trainval'  # 稍后进一步划分
    for file in test_files:
        file_split_map[file] = 'test'
    
    # 收集所有有效图像信息
    print("\n收集图像信息...")
    all_images = []
    missing_images = []
    
    # 创建breed_id到class_id的映射
    breed_id_to_class_id = {breed['breed_id']: breed['class_id'] for breed in breeds}
    breed_id_to_name = {breed['breed_id']: breed['breed_name'] for breed in breeds}
    
    total_checked = 0
    for breed in breeds:
        breed_folder = breed['folder_path']
        breed_id = breed['breed_id']
        breed_name = breed['breed_name']
        class_id = breed['class_id']
        
        print(f"处理品种: {breed_name}...")
        
        # 遍历该品种的所有图像
        for img_file in breed_folder.glob("*.jpg"):
            total_checked += 1
            
            # 构建相对路径（与MATLAB文件中的格式匹配）
            rel_path = f"{breed_folder.name}/{img_file.name}"
            
            # 检查图像是否损坏
            if not check_corrupted_images(img_file):
                print(f"  损坏的图像: {rel_path}")
                continue
            
            # 确定划分
            if rel_path in file_split_map:
                split = file_split_map[rel_path]
            else:
                # 如果没有在划分文件中，随机分配
                split = 'trainval' if random.random() < 0.8 else 'test'
            
            all_images.append({
                'rel_path': rel_path,
                'full_path': img_file,
                'breed_id': breed_id,
                'breed_name': breed_name,
                'class_id': class_id,
                'split': split
            })
    
    print(f"\n总共检查 {total_checked} 个图像文件")
    print(f"有效图像: {len(all_images)}")
    
    if len(all_images) == 0:
        print("错误: 没有找到有效的图像文件！")
        return
    
    # 将trainval进一步划分为train和val（8:2比例）
    train_images = []
    val_images = []
    test_images = []
    
    for img_info in all_images:
        if img_info['split'] == 'trainval':
            # 80%训练，20%验证
            if random.random() < 0.8:
                img_info['final_split'] = 'train'
                train_images.append(img_info)
            else:
                img_info['final_split'] = 'val'
                val_images.append(img_info)
        else:  # test
            img_info['final_split'] = 'test'
            test_images.append(img_info)
    
    print(f"\n最终划分:")
    print(f"  训练集: {len(train_images)} 张图片")
    print(f"  验证集: {len(val_images)} 张图片")
    print(f"  测试集: {len(test_images)} 张图片")
    
    # 处理并保存图像
    def process_and_save_images(image_list, split_name):
        """处理并保存图像到指定目录"""
        split_dir = processed_dir / split_name
        image_info = []
        
        for idx, img_info in enumerate(image_list):
            try:
                img_path = img_info['full_path']
                
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
                # 替换空格和特殊字符
                safe_breed_name = breed_name.replace(' ', '_').replace('/', '_')
                new_filename = f"{safe_breed_name}_{class_id:03d}_{split_name}_{idx:05d}.jpg"
                save_path = split_dir / new_filename
                
                # 保存图像
                cv2.imwrite(str(save_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                
                # 记录信息
                image_info.append({
                    'filename': new_filename,
                    'original_path': str(img_path),
                    'rel_path': img_info['rel_path'],
                    'breed_name': breed_name,
                    'class_id': class_id,
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
    test_info = process_and_save_images(test_images, 'test')
    
    # 合并所有信息
    all_info = train_info + val_info + test_info
    
    # 创建标签映射文件
    labels_data = []
    for breed in breeds:
        labels_data.append({
            'class_id': breed['class_id'],
            'breed_name': breed['breed_name'],
            'breed_id': breed['breed_id'],
            'folder_name': breed['folder_name']
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
        'image_size': config['image_size'],
        'breeds_count': len(breeds)
    }
    
    stats_path = processed_dir / "dataset_stats.yaml"
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    print("\n" + "="*50)
    print("预处理完成！")
    print(f"处理后的数据保存在: {processed_dir}")
    print(f"标签文件: {labels_path} (包含 {len(breeds)} 个品种)")
    print(f"图像信息文件: {info_path}")
    print(f"数据集统计: {stats_path}")
    print("="*50)
    
    # 显示统计信息
    print(f"\n总图像数: {len(all_info)}")
    print(f"品种数: {len(breeds)}")
    
    # 显示每个split的分布
    print("\n划分分布:")
    for split_name, split_info in [('train', train_info), ('val', val_info), ('test', test_info)]:
        if split_info:
            split_df = pd.DataFrame(split_info)
            print(f"\n{split_name} ({len(split_info)}张):")
            
            # 品种分布（前5个）
            breed_counts = split_df['breed_name'].value_counts().head(5)
            print(f"  前5个品种:")
            for breed, count in breed_counts.items():
                percentage = count / len(split_info) * 100
                print(f"    {breed:30s}: {count:3d} ({percentage:.1f}%)")
    
    # 显示整体品种分布（前10个）
    print("\n整体品种分布（前10个）:")
    all_df = pd.DataFrame(all_info)
    breed_counts = all_df['breed_name'].value_counts().head(10)
    for breed, count in breed_counts.items():
        percentage = count / len(all_info) * 100
        print(f"  {breed:30s}: {count:3d} ({percentage:.1f}%)")

if __name__ == "__main__":
    preprocess_stanford_dogs()