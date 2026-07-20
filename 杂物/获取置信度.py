#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理多个测试CSV文件，提取置信度统计数据
输出：TXT报告 + CSV文件（可直接导入Django Admin）
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import csv

# ==================== 配置区域 ====================
# 方式1：自动扫描模式（扫描指定目录下的所有CSV文件）
AUTO_SCAN_MODE = True  # 设为True启用自动扫描
SCAN_DIRECTORY = r"D:\\study\\graduation_cnn\\models\\vgg\\备份"  # 扫描目录

# 方式2：手动指定文件列表
CSV_FILE_LIST = [
    r"D:\study\graduation_cnn\models\vgg\备份\cats_vs_dogs\有数据增强\test_predictions_VGG16_cats_vs_dogs_20260224_194108.csv",
    
    # 可以继续添加更多文件
]

# 是否自动扫描其他模型（ResNet、MobileNet的CSV文件）
SCAN_OTHER_MODELS = True
# =================================================


def find_all_csv_files():
    """自动扫描所有CSV文件"""
    csv_files = []
    
    if AUTO_SCAN_MODE:
        for root, dirs, files in os.walk(SCAN_DIRECTORY):
            for file in files:
                if file.startswith('test_predictions_') and file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
    else:
        csv_files = CSV_FILE_LIST.copy()
        
        if SCAN_OTHER_MODELS:
            base_dirs = [
                r"D:\study\graduation_cnn\models\resnet",
                r"D:\study\graduation_cnn\models\mobilenet",
            ]
            for base_dir in base_dirs:
                if os.path.exists(base_dir):
                    for root, dirs, files in os.walk(base_dir):
                        for file in files:
                            if file.startswith('test_predictions_') and file.endswith('.csv'):
                                full_path = os.path.join(root, file)
                                if full_path not in csv_files:
                                    csv_files.append(full_path)
    
    return csv_files


def parse_filename_info(filepath):
    """从文件名解析信息"""
    filename = os.path.basename(filepath)
    
    # 解析模型名称
    if 'VGG16' in filename:
        model = 'vgg'
    elif 'ResNet' in filename or 'resnet' in filename.lower():
        model = 'resnet'
    elif 'MobileNet' in filename or 'mobilenet' in filename.lower():
        model = 'mobilenet'
    else:
        model = 'unknown'
    
    # 解析数据集
    dataset = 'unknown'
    if 'cats_vs_dogs' in filepath.lower():
        dataset = 'cats_vs_dogs'
    elif 'dogs77' in filepath.lower() or 'doges77' in filepath.lower():
        dataset = 'dogs77'
    elif 'oxford_pets' in filepath.lower():
        dataset = 'oxford_pets'
    
    # 判断任务类型
    if dataset == 'cats_vs_dogs':
        task = 'catdog'
    elif dataset in ['dogs77', 'oxford_pets']:
        task = 'dogbreed'
    else:
        task = 'unknown'
    
    # 判断是否使用数据增强
    augmentation = 'unknown'
    if '有数据增强' in filepath or 'augment' in filepath.lower():
        augmentation = 'yes'
    elif '无数据增强' in filepath or 'no_augment' in filepath.lower():
        augmentation = 'no'
    
    return model, task, dataset, augmentation


def load_csv_data(csv_path):
    """加载CSV文件"""
    df = pd.read_csv(csv_path)
    return df


def calculate_overall_stats(df):
    """计算整体统计"""
    correct_df = df[df['is_correct'] == True]
    wrong_df = df[df['is_correct'] == False]
    
    stats = {
        "total_samples": len(df),
        "correct_count": len(correct_df),
        "wrong_count": len(wrong_df),
        "accuracy": len(correct_df) / len(df) if len(df) > 0 else 0,
        "avg_confidence_all": df['confidence'].mean(),
        "avg_confidence_correct": correct_df['confidence'].mean() if len(correct_df) > 0 else 0,
        "avg_confidence_wrong": wrong_df['confidence'].mean() if len(wrong_df) > 0 else 0,
    }
    return stats


def calculate_by_class_stats(df):
    """按类别计算平均置信度"""
    class_stats = []
    
    for class_name in sorted(df['true_class'].unique()):
        class_df = df[df['true_class'] == class_name]
        correct_df = class_df[class_df['is_correct'] == True]
        
        stats = {
            "class_name": class_name,
            "sample_count": len(class_df),
            "correct_count": len(correct_df),
            "accuracy": len(correct_df) / len(class_df) if len(class_df) > 0 else 0,
            "avg_confidence": class_df['confidence'].mean(),
            "avg_confidence_correct": correct_df['confidence'].mean() if len(correct_df) > 0 else 0,
        }
        class_stats.append(stats)
    
    class_stats.sort(key=lambda x: x['class_name'])
    return class_stats


def save_txt_report(overall_stats, class_stats, output_path, file_info):
    """保存TXT报告"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"置信度统计报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"源文件: {file_info['filename']}")
    lines.append(f"模型: {file_info['model']} | 任务: {file_info['task']} | 数据集: {file_info['dataset']} | 数据增强: {file_info['augmentation']}")
    lines.append("=" * 80)
    lines.append("")
    
    # 整体统计
    lines.append("【整体统计】")
    lines.append("-" * 50)
    lines.append(f"总样本数:           {overall_stats['total_samples']}")
    lines.append(f"正确预测数:         {overall_stats['correct_count']}")
    lines.append(f"错误预测数:         {overall_stats['wrong_count']}")
    lines.append(f"准确率:             {overall_stats['accuracy']:.4f} ({overall_stats['accuracy']*100:.2f}%)")
    lines.append(f"整体平均置信度:     {overall_stats['avg_confidence_all']:.4f} ({overall_stats['avg_confidence_all']*100:.2f}%)")
    lines.append(f"正确预测平均置信度: {overall_stats['avg_confidence_correct']:.4f} ({overall_stats['avg_confidence_correct']*100:.2f}%)")
    lines.append(f"错误预测平均置信度: {overall_stats['avg_confidence_wrong']:.4f} ({overall_stats['avg_confidence_wrong']*100:.2f}%)")
    lines.append("")
    
    # 各类别统计
    lines.append("【各类别平均置信度】")
    lines.append("-" * 80)
    lines.append(f"{'类别名称':<35} {'样本数':<8} {'准确率':<12} {'平均置信度':<12}")
    lines.append("-" * 80)
    
    for stat in class_stats:
        class_name = stat['class_name'][:34] if len(stat['class_name']) > 34 else stat['class_name']
        lines.append(f"{class_name:<35} {stat['sample_count']:<8} "
                    f"{stat['accuracy']*100:>6.2f}%      "
                    f"{stat['avg_confidence']*100:>6.2f}%")
    
    lines.append("")
    lines.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def save_csv_files(overall_stats, class_stats, output_dir, file_info):
    """保存CSV文件（用于Django Admin导入）"""
    
    # 1. 保存模型整体统计CSV（追加模式，汇总所有文件）
    model_csv_path = os.path.join(output_dir, 'model_confidence_all.csv')
    model_exists = os.path.exists(model_csv_path)
    
    with open(model_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not model_exists:
            writer.writerow(['model_name', 'task', 'dataset', 'augmentation', 
                           'total_samples', 'accuracy', 'avg_confidence'])
        
        writer.writerow([
            file_info['model'],
            file_info['task'],
            file_info['dataset'],
            file_info['augmentation'],
            overall_stats['total_samples'],
            f"{overall_stats['accuracy']*100:.2f}",
            f"{overall_stats['avg_confidence_all']*100:.2f}"
        ])
    
    # 2. 保存各类别统计CSV（每个文件单独保存）
    class_csv_path = os.path.join(output_dir, f"class_confidence_{file_info['model']}_{file_info['task']}_{file_info['dataset']}.csv")
    
    with open(class_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'task', 'dataset', 'class_name', 
                        'sample_count', 'accuracy', 'avg_confidence'])
        
        for stat in class_stats:
            writer.writerow([
                file_info['model'],
                file_info['task'],
                file_info['dataset'],
                stat['class_name'],
                stat['sample_count'],
                f"{stat['accuracy']*100:.2f}",
                f"{stat['avg_confidence']*100:.2f}"
            ])
    
    return model_csv_path, class_csv_path


def process_single_file(csv_path, verbose=True):
    """处理单个CSV文件"""
    try:
        if verbose:
            print(f"\n📄 处理文件: {csv_path}")
        
        # 解析文件信息
        model, task, dataset, augmentation = parse_filename_info(csv_path)
        file_info = {
            'filename': os.path.basename(csv_path),
            'model': model,
            'task': task,
            'dataset': dataset,
            'augmentation': augmentation
        }
        
        # 加载数据
        df = load_csv_data(csv_path)
        
        # 计算统计
        overall_stats = calculate_overall_stats(df)
        class_stats = calculate_by_class_stats(df)
        
        # 输出目录（保存在CSV文件所在目录）
        output_dir = os.path.dirname(csv_path)
        
        # 保存TXT报告
        txt_path = os.path.join(output_dir, os.path.basename(csv_path).replace('.csv', '_confidence_stats.txt'))
        save_txt_report(overall_stats, class_stats, txt_path, file_info)
        
        # 保存CSV文件
        model_csv, class_csv = save_csv_files(overall_stats, class_stats, output_dir, file_info)
        
        if verbose:
            print(f"   ✅ 模型: {model}, 任务: {task}, 数据集: {dataset}")
            print(f"   📊 准确率: {overall_stats['accuracy']*100:.2f}%, 平均置信度: {overall_stats['avg_confidence_all']*100:.2f}%")
            print(f"   💾 TXT: {txt_path}")
            print(f"   💾 CSV: {class_csv}")
        
        return True, overall_stats, class_stats, file_info
        
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        return False, None, None, None


def main():
    import csv  # 确保csv模块已导入
    
    print("=" * 80)
    print("批量置信度统计工具（支持CSV导出）")
    print("=" * 80)
    
    # 获取所有CSV文件
    csv_files = find_all_csv_files()
    
    if not csv_files:
        print("❌ 未找到任何CSV文件")
        return 1
    
    print(f"\n📁 找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个文件
    success_count = 0
    results_summary = []
    
    for i, csv_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] 处理中...")
        success, overall, class_stats, file_info = process_single_file(csv_path, verbose=True)
        
        if success:
            success_count += 1
            results_summary.append({
                'file': csv_path,
                'model': file_info['model'],
                'task': file_info['task'],
                'dataset': file_info['dataset'],
                'augmentation': file_info['augmentation'],
                'accuracy': overall['accuracy'],
                'avg_confidence': overall['avg_confidence_all']
            })
    
    # 打印汇总报告
    print("\n" + "=" * 80)
    print("📊 处理完成汇总")
    print("=" * 80)
    print(f"成功处理: {success_count}/{len(csv_files)} 个文件")
    
    if results_summary:
        print("\n各文件结果摘要:")
        print("-" * 80)
        print(f"{'模型':<12} {'任务':<12} {'数据集':<15} {'数据增强':<8} {'准确率':<10} {'平均置信度':<10}")
        print("-" * 80)
        for r in results_summary:
            aug_text = "有增强" if r['augmentation'] == 'yes' else ("无增强" if r['augmentation'] == 'no' else "未知")
            print(f"{r['model']:<12} {r['task']:<12} {r['dataset']:<15} {aug_text:<8} "
                  f"{r['accuracy']*100:>6.2f}%     {r['avg_confidence']*100:>6.2f}%")
        
        # 提示CSV文件位置
        print("\n" + "=" * 80)
        print("📁 生成的CSV文件位置:")
        print("=" * 80)
        print("每个CSV文件所在目录下生成了：")
        print("  - *_confidence_stats.txt (详细报告)")
        print("  - class_confidence_*.csv (各类别统计，可直接导入Admin)")
        print("  - model_confidence_all.csv (所有模型的整体统计汇总)")
    
    print("\n✅ 处理完成")
    return 0


if __name__ == "__main__":
    exit(main())