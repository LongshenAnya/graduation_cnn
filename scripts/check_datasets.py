"""
数据集检查脚本
用于验证三个预处理后的数据集是否完整可用
"""

import os
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class DatasetChecker:
    """数据集检查器"""
    
    def __init__(self, base_dir="data/processed"):
        self.base_dir = Path(base_dir)
        self.datasets = {}
        self.results = {}
        
    def load_dataset_info(self, dataset_name):
        """加载数据集信息"""
        dataset_dir = self.base_dir / dataset_name
        
        if not dataset_dir.exists():
            print(f"❌ 数据集目录不存在: {dataset_dir}")
            return None
        
        # 加载统计文件
        stats_file = dataset_dir / "dataset_stats.yaml"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = yaml.safe_load(f)
        else:
            stats = {}
        
        # 加载标签文件
        labels_file = dataset_dir / "labels.csv"
        if labels_file.exists():
            labels_df = pd.read_csv(labels_file)
        else:
            labels_df = None
        
        # 加载图像信息文件
        info_file = dataset_dir / "images_info.csv"
        if info_file.exists():
            info_df = pd.read_csv(info_file)
        else:
            info_df = None
        
        # 检查各个子目录
        splits = {}
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                image_files = list(split_dir.glob("*.jpg"))
                splits[split] = {
                    'path': split_dir,
                    'count': len(image_files),
                    'files': image_files[:10]  # 只保留前10个文件路径用于检查
                }
        
        dataset_info = {
            'name': dataset_name,
            'path': dataset_dir,
            'stats': stats,
            'labels': labels_df,
            'info': info_df,
            'splits': splits,
            'has_labels': labels_df is not None,
            'has_info': info_df is not None,
            'has_stats': bool(stats)
        }
        
        self.datasets[dataset_name] = dataset_info
        return dataset_info
    
    def check_all_datasets(self):
        """检查所有数据集"""
        dataset_names = ['cats_vs_dogs', 'oxford_pets', 'stanford_dogs']
        
        print("=" * 60)
        print("开始检查数据集完整性")
        print("=" * 60)
        
        for dataset_name in dataset_names:
            print(f"\n📁 检查数据集: {dataset_name}")
            print("-" * 40)
            
            dataset_info = self.load_dataset_info(dataset_name)
            if dataset_info is None:
                self.results[dataset_name] = {'status': 'missing', 'errors': ['目录不存在']}
                continue
            
            # 执行各项检查
            checks = self.perform_checks(dataset_info)
            status = 'complete' if all(checks.values()) else 'partial'
            
            if all(checks.values()):
                print(f"✅ {dataset_name} 数据集完整")
            else:
                print(f"⚠️  {dataset_name} 数据集不完整")
            
            self.results[dataset_name] = {
                'status': status,
                'checks': checks,
                'info': dataset_info
            }
        
        return self.results
    
    def perform_checks(self, dataset_info):
        """执行具体的检查项"""
        checks = {
            'directory_exists': False,
            'splits_exist': False,
            'labels_exist': False,
            'info_exist': False,
            'stats_exist': False,
            'image_readable': False,
            'class_distribution': False
        }
        
        # 检查1: 目录是否存在
        if dataset_info['path'].exists():
            checks['directory_exists'] = True
            print(f"  ✓ 目录存在: {dataset_info['path']}")
        else:
            print(f"  ✗ 目录不存在: {dataset_info['path']}")
        
        # 检查2: 划分目录是否存在
        splits = dataset_info['splits']
        if splits:
            checks['splits_exist'] = True
            print(f"  ✓ 找到 {len(splits)} 个划分:")
            for split_name, split_info in splits.items():
                print(f"    - {split_name}: {split_info['count']} 张图片")
        else:
            print("  ✗ 未找到任何划分目录")
        
        # 检查3: 标签文件是否存在
        if dataset_info['has_labels']:
            checks['labels_exist'] = True
            num_classes = len(dataset_info['labels'])
            print(f"  ✓ 标签文件: {num_classes} 个类别")
        else:
            print("  ✗ 标签文件缺失")
        
        # 检查4: 图像信息文件是否存在
        if dataset_info['has_info']:
            checks['info_exist'] = True
            num_images = len(dataset_info['info'])
            print(f"  ✓ 图像信息文件: {num_images} 条记录")
        else:
            print("  ✗ 图像信息文件缺失")
        
        # 检查5: 统计文件是否存在
        if dataset_info['has_stats']:
            checks['stats_exist'] = True
            stats = dataset_info['stats']
            print(f"  ✓ 统计文件: {stats.get('total_images', 'N/A')} 张图片")
        else:
            print("  ✗ 统计文件缺失")
        
        # 检查6: 随机检查几张图像是否可读
        if splits:
            readable_count = 0
            total_checked = 0
            
            for split_name, split_info in splits.items():
                if split_info['files']:
                    for img_path in split_info['files'][:3]:  # 每个划分检查3张
                        total_checked += 1
                        try:
                            img = Image.open(img_path)
                            img.verify()
                            readable_count += 1
                        except:
                            pass
            
            if total_checked > 0:
                checks['image_readable'] = (readable_count == total_checked)
                if checks['image_readable']:
                    print(f"  ✓ 图像可读性: {readable_count}/{total_checked} 张图像可读")
                else:
                    print(f"  ✗ 图像可读性: 只有 {readable_count}/{total_checked} 张图像可读")
        
        # 检查7: 类别分布（如果有信息文件）
        if dataset_info['has_info']:
            info_df = dataset_info['info']
            if 'class_id' in info_df.columns:
                checks['class_distribution'] = True
                
                # 检查是否有NaN值
                nan_count = info_df['class_id'].isna().sum()
                if nan_count == 0:
                    print(f"  ✓ 类别分布: 无缺失值")
                else:
                    print(f"  ✗ 类别分布: 有 {nan_count} 个缺失值")
            else:
                print("  ✗ 类别分布: 信息文件中缺少class_id列")
        
        return checks
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "=" * 60)
        print("数据集检查总结报告")
        print("=" * 60)
        
        summary_data = []
        
        for dataset_name, result in self.results.items():
            if result['status'] == 'missing':
                summary_data.append([dataset_name, "❌ 缺失", "0", "0", "0", "N/A"])
                continue
            
            info = result['info']
            stats = info['stats']
            splits = info['splits']
            
            # 获取各类统计
            total_images = stats.get('total_images', 0)
            num_classes = len(info['labels']) if info['has_labels'] else 0
            
            # 获取各划分数量
            train_count = splits.get('train', {}).get('count', 0)
            val_count = splits.get('val', {}).get('count', 0)
            test_count = splits.get('test', {}).get('count', 0)
            
            # 计算检查通过率
            checks = result['checks']
            passed = sum(checks.values())
            total = len(checks)
            pass_rate = f"{passed}/{total}"
            
            status_icon = "✅" if result['status'] == 'complete' else "⚠️ "
            
            summary_data.append([
                dataset_name,
                f"{status_icon} {result['status']}",
                f"{total_images}",
                f"{num_classes}",
                f"{train_count}/{val_count}/{test_count}",
                pass_rate
            ])
        
        # 打印表格
        headers = ["数据集", "状态", "总图像数", "类别数", "划分(train/val/test)", "检查通过"]
        col_widths = [15, 12, 10, 10, 20, 12]
        
        # 打印表头
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = "-" * len(header_line)
        print(separator)
        print(header_line)
        print(separator)
        
        # 打印数据行
        for row in summary_data:
            row_line = " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
            print(row_line)
        
        print(separator)
        
        # 打印详细问题
        print("\n🔍 详细问题:")
        has_issues = False
        for dataset_name, result in self.results.items():
            if result['status'] != 'complete':
                has_issues = True
                print(f"\n{dataset_name}:")
                checks = result['checks']
                for check_name, check_passed in checks.items():
                    if not check_passed:
                        print(f"  ✗ {check_name}")
        
        if not has_issues:
            print("  ✓ 所有数据集均通过检查！")
    
    def visualize_dataset_stats(self):
        """可视化数据集统计信息"""
        if not self.datasets:
            print("未加载任何数据集信息")
            return
        
        n_datasets = len(self.datasets)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('数据集统计分析', fontsize=16, fontweight='bold')
        
        # 1. 数据集大小对比
        ax1 = axes[0, 0]
        dataset_names = []
        dataset_sizes = []
        dataset_colors = []
        
        color_map = {
            'cats_vs_dogs': '#FF6B6B',
            'oxford_pets': '#4ECDC4',
            'stanford_dogs': '#45B7D1'
        }
        
        for dataset_name, info in self.datasets.items():
            if info and 'stats' in info and info['stats']:
                dataset_names.append(dataset_name)
                dataset_sizes.append(info['stats'].get('total_images', 0))
                dataset_colors.append(color_map.get(dataset_name, '#95A5A6'))
        
        bars = ax1.bar(dataset_names, dataset_sizes, color=dataset_colors, alpha=0.8)
        ax1.set_title('各数据集图像数量', fontsize=12)
        ax1.set_ylabel('图像数量')
        ax1.set_xlabel('数据集')
        
        # 在柱子上添加数值
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        # 2. 类别数量对比
        ax2 = axes[0, 1]
        class_counts = []
        
        for dataset_name in dataset_names:
            info = self.datasets[dataset_name]
            if info['has_labels']:
                class_counts.append(len(info['labels']))
            else:
                class_counts.append(0)
        
        bars2 = ax2.bar(dataset_names, class_counts, color=dataset_colors, alpha=0.8)
        ax2.set_title('各数据集类别数量', fontsize=12)
        ax2.set_ylabel('类别数量')
        ax2.set_xlabel('数据集')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        # 3. 划分比例（饼图）
        ax3 = axes[1, 0]
        
        # 使用第一个数据集作为示例
        if dataset_names:
            example_dataset = dataset_names[0]
            info = self.datasets[example_dataset]
            
            if info['splits']:
                split_labels = []
                split_sizes = []
                split_colors = ['#FF9F1C', '#2EC4B6', '#E71D36']
                
                for idx, (split_name, split_info) in enumerate(info['splits'].items()):
                    split_labels.append(split_name)
                    split_sizes.append(split_info['count'])
                
                wedges, texts, autotexts = ax3.pie(split_sizes, labels=split_labels, 
                                                  colors=split_colors[:len(split_labels)],
                                                  autopct='%1.1f%%', startangle=90)
                ax3.set_title(f'{example_dataset} 数据集划分比例', fontsize=12)
                ax3.axis('equal')
        
        # 4. 类别分布示例
        ax4 = axes[1, 1]
        
        # 显示一个数据集的类别分布（前10个类别）
        for dataset_name in dataset_names:
            info = self.datasets[dataset_name]
            if info['has_info'] and 'class_id' in info['info'].columns:
                # 获取前10个最常见的类别
                class_dist = info['info']['class_id'].value_counts().head(10)
                
                if len(class_dist) > 0:
                    # 获取类别名称
                    if info['has_labels']:
                        labels_df = info['labels']
                        class_names = {}
                        for _, row in labels_df.iterrows():
                            if 'class_id' in row and 'breed_name' in row:
                                class_names[row['class_id']] = row['breed_name']
                            elif 'class_id' in row and 'class_name' in row:
                                class_names[row['class_id']] = row['class_name']
                        
                        # 创建带有类别名的标签
                        x_labels = [class_names.get(cls_id, f'Class {cls_id}') 
                                   for cls_id in class_dist.index]
                    else:
                        x_labels = [f'Class {cls_id}' for cls_id in class_dist.index]
                    
                    ax4.barh(x_labels, class_dist.values, alpha=0.7)
                    ax4.set_title(f'{dataset_name} 类别分布 (前10)', fontsize=12)
                    ax4.set_xlabel('图像数量')
                    break
        
        plt.tight_layout()
        plt.savefig('data/dataset_stats_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 统计图表已保存到: data/dataset_stats_visualization.png")
        plt.show()
    
    def check_image_samples(self):
        """随机检查一些图像样本"""
        print("\n" + "=" * 60)
        print("图像样本检查")
        print("=" * 60)
        
        for dataset_name, info in self.datasets.items():
            if not info or not info['splits']:
                continue
            
            print(f"\n📸 {dataset_name}:")
            
            # 从每个划分中随机选择一张图像
            for split_name, split_info in info['splits'].items():
                if split_info['files']:
                    # 随机选择一个图像
                    import random
                    sample_image = random.choice(split_info['files'])
                    
                    try:
                        # 读取图像信息
                        img = Image.open(sample_image)
                        
                        # 查找对应的标签信息
                        if info['has_info']:
                            img_filename = sample_image.name
                            img_info = info['info'][info['info']['filename'] == img_filename]
                            
                            if not img_info.empty:
                                class_id = img_info.iloc[0]['class_id']
                                if info['has_labels']:
                                    label_row = info['labels'][info['labels']['class_id'] == class_id]
                                    if not label_row.empty:
                                        class_name = label_row.iloc[0].get('breed_name') or label_row.iloc[0].get('class_name', f'Class {class_id}')
                                    else:
                                        class_name = f'Class {class_id}'
                                else:
                                    class_name = f'Class {class_id}'
                            else:
                                class_name = "Unknown"
                        else:
                            class_name = "Unknown"
                        
                        print(f"  {split_name}: {sample_image.name}")
                        print(f"    尺寸: {img.size}, 模式: {img.mode}")
                        print(f"    类别: {class_name}")
                        
                        # 显示缩略信息
                        if split_name == 'train':  # 只显示一个示例
                            print(f"    路径: {sample_image}")
                            print()
                    
                    except Exception as e:
                        print(f"  {split_name}: 无法读取图像 {sample_image.name} - {e}")
    
    def save_report(self, output_file="data/dataset_validation_report.md"):
        """保存详细报告到Markdown文件"""
        report_lines = []
        
        report_lines.append("# 数据集验证报告")
        report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for dataset_name, result in self.results.items():
            report_lines.append(f"## {dataset_name}")
            report_lines.append("")
            
            if result['status'] == 'missing':
                report_lines.append("❌ **数据集缺失**")
                report_lines.append("")
                continue
            
            info = result['info']
            stats = info['stats']
            
            # 基本信息
            report_lines.append("### 基本信息")
            report_lines.append(f"- **状态**: {'✅ 完整' if result['status'] == 'complete' else '⚠️ 不完整'}")
            report_lines.append(f"- **总图像数**: {stats.get('total_images', 'N/A')}")
            report_lines.append(f"- **类别数**: {len(info['labels']) if info['has_labels'] else 'N/A'}")
            report_lines.append(f"- **图像尺寸**: {stats.get('image_size', 'N/A')}")
            report_lines.append("")
            
            # 划分信息
            report_lines.append("### 数据划分")
            for split_name, split_info in info['splits'].items():
                report_lines.append(f"- **{split_name}**: {split_info['count']} 张图像")
            report_lines.append("")
            
            # 检查结果
            report_lines.append("### 完整性检查")
            checks = result['checks']
            for check_name, check_passed in checks.items():
                status = "✅" if check_passed else "❌"
                report_lines.append(f"- {status} {check_name}")
            report_lines.append("")
            
            # 类别信息（如果有）
            if info['has_labels'] and len(info['labels']) <= 20:  # 只显示小数据集的完整类别
                report_lines.append("### 类别列表")
                for _, row in info['labels'].iterrows():
                    class_id = row.get('class_id', '')
                    class_name = row.get('breed_name') or row.get('class_name', '')
                    report_lines.append(f"- {class_id}: {class_name}")
                report_lines.append("")
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n📄 详细报告已保存到: {output_file}")

def main():
    """主函数"""
    print("正在检查数据集完整性...\n")
    
    checker = DatasetChecker()
    
    # 1. 检查所有数据集
    results = checker.check_all_datasets()
    
    # 2. 生成总结报告
    checker.generate_summary_report()
    
    # 3. 检查图像样本
    checker.check_image_samples()
    
    # 4. 可视化统计
    try:
        checker.visualize_dataset_stats()
    except Exception as e:
        print(f"\n⚠️  可视化生成失败: {e}")
        print("请确保已安装matplotlib和seaborn: pip install matplotlib seaborn")
    
    # 5. 保存详细报告
    checker.save_report()
    
    print("\n" + "=" * 60)
    print("✅ 数据集检查完成！")
    print("=" * 60)
    
    # 返回检查结果
    all_complete = all(r['status'] == 'complete' for r in results.values())
    return all_complete

if __name__ == "__main__":
    # 运行检查
    success = main()
    
    if success:
        print("\n🎉 所有数据集均完整可用，可以开始训练模型！")
        print("\n下一步建议:")
        print("1. 创建统一的数据加载器 (dataset_loader.py)")
        print("2. 开始构建和训练CNN模型")
        print("3. 设计模型对比实验")
    else:
        print("\n⚠️  部分数据集存在问题，请检查并修复后再继续。")
        print("\n常见问题解决:")
        print("1. 确保数据集已正确预处理")
        print("2. 检查文件路径和权限")
        print("3. 重新运行预处理脚本")
    
    exit(0 if success else 1)