import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def calculate_metrics_from_csv(csv_path, output_dir='results/metrics'):
    """
    从测试结果 CSV 文件计算分类指标
    
    参数:
        csv_path: str，CSV 文件路径
        output_dir: str，输出目录
    
    返回:
        metrics_df: DataFrame，各类别的指标
        overall_df: DataFrame，整体指标
    """
    
    # ===== 1. 读取数据 =====
    df = pd.read_csv(csv_path)
    
    # 检查必要列
    required_cols = ['true_label', 'pred_label', 'true_class', 'pred_class']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 文件缺少必要列: {col}")
    
    # 提取真实标签和预测标签
    y_true = df['true_label'].values
    y_pred = df['pred_label'].values
    
    # 获取类别名称映射（从 true_class 和 true_label 的唯一对应关系）
    class_mapping = df[['true_label', 'true_class']].drop_duplicates().sort_values('true_label')
    class_names = class_mapping['true_class'].values
    n_classes = len(class_names)
    
    print(f"\n{'='*60}")
    print(f"文件: {os.path.basename(csv_path)}")
    print(f"总样本数: {len(df)}")
    print(f"类别数: {n_classes}")
    print(f"{'='*60}")
    
    # ===== 2. 计算整体指标 =====
    # 宏平均（所有类别权重相同）
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 微平均（按样本数加权）
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # 加权平均（按各类别样本数加权）
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 准确率
    accuracy = (y_true == y_pred).sum() / len(y_true)
    
    overall_metrics = {
        '指标类型': ['宏平均 (Macro)', '微平均 (Micro)', '加权平均 (Weighted)'],
        '精确率 (%)': [macro_precision * 100, micro_precision * 100, weighted_precision * 100],
        '召回率 (%)': [macro_recall * 100, micro_recall * 100, weighted_recall * 100],
        'F1分数 (%)': [macro_f1 * 100, micro_f1 * 100, weighted_f1 * 100],
        '准确率 (%)': [accuracy * 100, accuracy * 100, accuracy * 100]
    }
    overall_df = pd.DataFrame(overall_metrics)
    
    # ===== 3. 计算每个类别的指标 =====
    # 计算混淆矩阵相关统计
    per_class_metrics = []
    
    for label in range(n_classes):
        # TP: 预测为该类别且真实为该类别
        tp = ((y_true == label) & (y_pred == label)).sum()
        # FP: 预测为该类别但真实不是
        fp = ((y_true != label) & (y_pred == label)).sum()
        # FN: 真实为该类别但预测不是
        fn = ((y_true == label) & (y_pred != label)).sum()
        # TN: 真实不是且预测也不是
        tn = ((y_true != label) & (y_pred != label)).sum()
        
        # 样本数
        support = (y_true == label).sum()
        
        # 精确率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # 召回率
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # F1 分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics.append({
            '类别ID': label,
            '类别名称': class_names[label],
            '样本数': support,
            '精确率 (%)': precision * 100,
            '召回率 (%)': recall * 100,
            'F1分数 (%)': f1 * 100,
            'TP': tp,
            'FP': fp,
            'FN': fn
        })
    
    per_class_df = pd.DataFrame(per_class_metrics)
    
    # ===== 4. 打印结果 =====
    print(f"\n{'='*80}")
    print("整体指标汇总")
    print(f"{'='*80}")
    print(overall_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("各类别详细指标（按 F1 分数降序排列）")
    print(f"{'='*80}")
    
    # 按 F1 分数降序排列
    per_class_sorted = per_class_df.sort_values('F1分数 (%)', ascending=False)
    
    # 只显示关键列
    display_cols = ['类别名称', '样本数', '精确率 (%)', '召回率 (%)', 'F1分数 (%)']
    print(per_class_sorted[display_cols].to_string(index=False))
    
    # ===== 5. 找出表现最好和最差的类别 =====
    best_class = per_class_df.loc[per_class_df['F1分数 (%)'].idxmax()]
    worst_class = per_class_df.loc[per_class_df['F1分数 (%)'].idxmin()]
    
    print(f"\n{'='*80}")
    print("最佳与最差表现类别")
    print(f"{'='*80}")
    print(f"最佳: {best_class['类别名称']} (F1={best_class['F1分数 (%)']:.2f}%, "
          f"精确率={best_class['精确率 (%)']:.2f}%, 召回率={best_class['召回率 (%)']:.2f}%)")
    print(f"最差: {worst_class['类别名称']} (F1={worst_class['F1分数 (%)']:.2f}%, "
          f"精确率={worst_class['精确率 (%)']:.2f}%, 召回率={worst_class['召回率 (%)']:.2f}%)")
    
    # ===== 6. 保存结果 =====
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 从文件名提取模型和数据集信息
        basename = os.path.basename(csv_path).replace('.csv', '')
        
        # 保存整体指标
        overall_path = f'{output_dir}/{basename}_overall_metrics.csv'
        overall_df.to_csv(overall_path, index=False, encoding='utf-8-sig')
        
        # 保存各类别指标
        per_class_path = f'{output_dir}/{basename}_per_class_metrics.csv'
        per_class_df.to_csv(per_class_path, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*80}")
        print(f"结果已保存到: {output_dir}")
        print(f"整体指标: {os.path.basename(overall_path)}")
        print(f"各类别指标: {os.path.basename(per_class_path)}")
        print(f"{'='*80}")
    
    return per_class_df, overall_df


def calculate_summary_table(csv_files, model_names=None, output_dir='results/metrics'):
    """
    汇总多个模型测试结果的对比表格（包含精确率、召回率、F1分数）
    
    参数:
        csv_files: list，CSV 文件路径列表
        model_names: list，模型名称列表（默认从文件名提取）
        output_dir: str，输出目录
    
    返回:
        summary_df: DataFrame，汇总对比表
    """
    
    if model_names is None:
        model_names = []
        for path in csv_files:
            basename = os.path.basename(path)
            if 'VGG' in basename:
                model_names.append('VGG16')
            elif 'ResNet' in basename:
                model_names.append('ResNet34')
            elif 'MobileNet' in basename:
                model_names.append('MobileNet')
            else:
                model_names.append(basename[:20])
    
    summary_data = []
    
    for path, name in zip(csv_files, model_names):
        df = pd.read_csv(path)
        y_true = df['true_label'].values
        y_pred = df['pred_label'].values
        
        # 计算各项指标（加权平均）
        accuracy = (y_true == y_pred).sum() / len(y_true) * 100
        
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        
        summary_data.append({
            '模型': name,
            '样本数': len(df),
            '准确率 (%)': round(accuracy, 2),
            '宏平均 精确率 (%)': round(precision_macro, 2),
            '宏平均 召回率 (%)': round(recall_macro, 2),
            '宏平均 F1 (%)': round(f1_macro, 2),
            '加权平均 精确率 (%)': round(precision_weighted, 2),
            '加权平均 召回率 (%)': round(recall_weighted, 2),
            '加权平均 F1 (%)': round(f1_weighted, 2)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n{'='*100}")
    print("模型测试结果汇总对比")
    print(f"{'='*100}")
    print(summary_df.to_string(index=False))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = f'{output_dir}/model_comparison_summary.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总表已保存到: {summary_path}")
    
    return summary_df

def extract_model_dataset_name(csv_path):
    """
    从 CSV 文件名提取模型和数据集信息
    
    示例: test_predictions_VGG16_oxford_pets_20260306_170216.csv
    -> 模型: VGG16, 数据集: oxford_pets
    """
    basename = os.path.basename(csv_path)
    parts = basename.replace('.csv', '').split('_')
    
    model = None
    dataset = None
    
    # 常见模型名称
    model_keywords = ['VGG16', 'VGG', 'ResNet34', 'ResNet', 'MobileNet', 'Mobile']
    dataset_keywords = ['cats', 'dogs', 'oxford', 'pets', 'stanford', 'breeds']
    
    for part in parts:
        for kw in model_keywords:
            if kw.lower() in part.lower():
                model = part
                break
        for kw in dataset_keywords:
            if kw.lower() in part.lower():
                if dataset is None:
                    dataset = part
                else:
                    dataset = f"{dataset}_{part}"
    
    return model, dataset


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    # 示例1：处理单个 CSV 文件
    csv_path = r"D:\study\graduation_cnn\models\mobilenet\备份\oxford_pets\test_predictions_MobileNetV1_oxford_pets_20260418_211754.csv"

    # 提取模型和数据集名称
    model, dataset = extract_model_dataset_name(csv_path)
    print(f"模型: {model}, 数据集: {dataset}")
    
    # 计算指标
    per_class_df, overall_df = calculate_metrics_from_csv(
        csv_path, 
        output_dir='D:\\study\\graduation_cnn\\models\\mobilenet\\备份\\oxford_pets'
    )
    
    # 示例2：批量处理多个 CSV 文件并汇总对比
    # csv_files = [
    #     'test_predictions_VGG16_cats_vs_dogs.csv',
    #     'test_predictions_ResNet34_cats_vs_dogs.csv',
    #     'test_predictions_MobileNet_cats_vs_dogs.csv'
    # ]
    # summary_df = calculate_summary_table(csv_files, output_dir='results/metrics')