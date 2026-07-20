import torch
import torch.utils.data as Data
from torchvision import transforms
import pandas as pd
import os
import datetime
import yaml
import time  # 新增：用于时间统计
from dataloader import get_dataloaders
from model import VGG16

def load_class_names(dataset_path):
    """从数据集目录加载类别名称"""
    labels_csv_path = os.path.join(dataset_path, 'labels.csv')
    
    if os.path.exists(labels_csv_path):
        try:
            labels_df = pd.read_csv(labels_csv_path)
            if 'class_id' in labels_df.columns:
                labels_df = labels_df.sort_values('class_id')
                
                # 处理不同的列名
                if 'class_name' in labels_df.columns:
                    class_names = labels_df['class_name'].tolist()
                    print(f"从 {labels_csv_path} 加载了 {len(class_names)} 个类别名称 (class_name列)")
                elif 'breed_name' in labels_df.columns:
                    class_names = labels_df['breed_name'].tolist()
                    print(f"从 {labels_csv_path} 加载了 {len(class_names)} 个品种名称 (breed_name列)")
                else:
                    # 如果没有名称列，使用class_id作为名称
                    class_names = [f"Class_{i}" for i in range(len(labels_df))]
                    print(f"警告: 未找到类别名称列，使用默认名称")
                
                return class_names
        except Exception as e:
            print(f"读取 {labels_csv_path} 时出错: {e}")
    
    # 尝试从stats文件加载
    stats_yaml_path = os.path.join(dataset_path, 'dataset_stats.yaml')
    if os.path.exists(stats_yaml_path):
        try:
            with open(stats_yaml_path, 'r', encoding='utf-8') as f:
                stats = yaml.safe_load(f)
                if 'classes' in stats:
                    if isinstance(stats['classes'], list):
                        return stats['classes']
                    elif isinstance(stats['classes'], dict):
                        sorted_classes = dict(sorted(stats['classes'].items()))
                        return list(sorted_classes.values())
        except Exception as e:
            print(f"读取 {stats_yaml_path} 时出错: {e}")
    
    return None

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"

def save_results_to_txt(test_acc, confusion_matrix, class_names, filename, dataset_name, 
                       predictions_csv_path, total_time, avg_time_per_image, total_images):
    """保存测试结果到txt文件，包含时间统计"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"模型测试结果 - {dataset_name}\n")
        f.write("=" * 60 + "\n")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"类别数量: {len(class_names)}\n")
        f.write(f"总样本数: {total_images}\n")
        f.write(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
        f.write(f"总花费时间: {format_time(total_time)}\n")
        f.write(f"平均每张图片时间: {avg_time_per_image*1000:.2f}毫秒\n")
        f.write(f"详细预测结果: {os.path.basename(predictions_csv_path)}\n")
        f.write("\n")
        
        # 类别名称对应
        f.write("类别名称对应:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
        f.write("\n")
        
        # 混淆矩阵
        f.write("混淆矩阵:\n")
        f.write("行: 真实标签, 列: 预测标签\n\n")
        
        # 表头
        header = "True\\Pred" + "".join([f"{name[:8]:>10}" for name in class_names])
        f.write(header + "\n")
        f.write("-" * (10 + 10 * len(class_names)) + "\n")
        
        # 矩阵内容
        for i in range(len(class_names)):
            row = f"{class_names[i][:8]:<10}"
            for j in range(len(class_names)):
                row += f"{confusion_matrix[i][j]:>10}"
            f.write(row + "\n")
        
        # 各类别详细统计
        f.write("\n各类别详细统计:\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(class_names)):
            total = sum(confusion_matrix[i])
            correct = confusion_matrix[i][i]
            
            if total > 0:
                recall = correct / total
                pred_total = sum(confusion_matrix[j][i] for j in range(len(class_names)))
                precision = correct / pred_total if pred_total > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"{class_names[i]}:\n")
                f.write(f"  样本数: {total}\n")
                f.write(f"  正确数: {correct}\n")
                f.write(f"  准确率: {recall:.4f} ({recall*100:.2f}%)\n")
                f.write(f"  精确率: {precision:.4f} ({precision*100:.2f}%)\n")
                f.write(f"  F1分数: {f1:.4f}\n")
                
                errors = []
                for j in range(len(class_names)):
                    if i != j and confusion_matrix[i][j] > 0:
                        errors.append(f"{class_names[j]}: {confusion_matrix[i][j]}")
                
                if errors:
                    f.write(f"  误判为: {', '.join(errors)}\n")
                f.write("\n")
        
        # 总体统计
        f.write("总体统计:\n")
        f.write("-" * 30 + "\n")
        total_samples = sum(sum(row) for row in confusion_matrix)
        total_correct = sum(confusion_matrix[i][i] for i in range(len(class_names)))
        
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"正确预测数: {total_correct}\n")
        f.write(f"错误预测数: {total_samples - total_correct}\n")
        f.write(f"总体准确率: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
        
        # 时间性能统计
        f.write("\n时间性能统计:\n")
        f.write("-" * 30 + "\n")
        f.write(f"总推理时间: {format_time(total_time)}\n")
        f.write(f"平均每张图片: {avg_time_per_image*1000:.2f}毫秒\n")
        f.write(f"每秒处理图片数: {total_images/total_time:.2f} 张/秒\n")
        
        # 计算宏平均F1
        macro_f1 = 0
        valid_classes = 0
        for i in range(len(class_names)):
            total = sum(confusion_matrix[i])
            correct = confusion_matrix[i][i]
            if total > 0:
                recall = correct / total
                pred_total = sum(confusion_matrix[j][i] for j in range(len(class_names)))
                precision = correct / pred_total if pred_total > 0 else 0
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    macro_f1 += f1
                    valid_classes += 1
        
        if valid_classes > 0:
            macro_f1 /= valid_classes
            f.write(f"宏平均F1分数: {macro_f1:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")

def save_predictions_to_csv(all_predictions, filename):
    """保存所有预测结果到CSV文件"""
    df = pd.DataFrame(all_predictions)
    df.to_csv(filename, index=False, encoding='utf-8')
    return filename

def test_model(model, test_loader, class_names, dataset_name, save_dir='models/vgg/results'):
    """
    测试模型并保存结果
    
    参数:
        model: 要测试的模型
        test_loader: 测试数据加载器
        class_names: 类别名称列表
        dataset_name: 数据集名称
        save_dir: 结果保存目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    test_corrects = 0.0
    test_num = 0
    
    # 初始化混淆矩阵和预测结果列表
    num_classes = len(class_names)
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    all_predictions = []
    
    # 记录开始时间
    total_start_time = time.time()
    batch_times = []  # 记录每个batch的时间
    
    with torch.no_grad():
        batch_idx = 0
        for test_inputs, test_labels in test_loader:
            # 记录batch开始时间
            batch_start_time = time.time()
            
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            
            # 模型设为评估模式
            model.eval()

            # 前向传播
            outputs = model(test_inputs)
            
            # 预测结果
            pre_lab = torch.argmax(outputs, dim=1)

            # 统计测试集准确率
            batch_corrects = torch.sum(pre_lab == test_labels)
            test_corrects += batch_corrects
            test_num += test_labels.size(0)
            
            # 更新混淆矩阵
            for i in range(len(test_labels)):
                true_label = test_labels[i].item()
                pred_label = pre_lab[i].item()
                confusion_matrix[true_label][pred_label] += 1
            
            # 收集预测结果
            for i in range(len(test_labels)):
                all_predictions.append({
                    'batch_id': batch_idx,
                    'sample_id_in_batch': i,
                    'true_label': test_labels[i].item(),
                    'pred_label': pre_lab[i].item(),
                    'true_class': class_names[test_labels[i].item()],
                    'pred_class': class_names[pre_lab[i].item()],
                    'is_correct': (test_labels[i] == pre_lab[i]).item(),
                    'confidence': torch.softmax(outputs[i], dim=0)[pre_lab[i].item()].item()
                })
            
            # 记录batch结束时间
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            batch_idx += 1
            
            # 每50个batch显示一次进度和时间
            if batch_idx % 50 == 0:
                avg_batch_time = sum(batch_times[-50:]) / min(50, len(batch_times))
                estimated_remaining = avg_batch_time * (len(test_loader) - batch_idx)
                print(f"  已处理 {batch_idx}/{len(test_loader)} 批次 | "
                      f"批次平均时间: {avg_batch_time:.3f}秒 | "
                      f"预计剩余: {format_time(estimated_remaining)}")
    
    # 计算总时间和平均时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    avg_time_per_image = total_time / test_num if test_num > 0 else 0
    
    test_acc = test_corrects.double().item() / test_num
    
    print(f"\n测试完成!")
    print(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"测试样本数: {test_num}")
    print(f"正确预测数: {int(test_corrects)}")
    print(f"总花费时间: {format_time(total_time)}")
    print(f"平均每张图片: {avg_time_per_image*1000:.2f}毫秒")
    print(f"处理速度: {test_num/total_time:.2f} 张/秒")
    
    # 生成保存文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.__class__.__name__
    
    # 1. 保存CSV预测结果
    csv_filename = f'{save_dir}/test_predictions_{model_name}_{dataset_name}_{timestamp}.csv'
    save_predictions_to_csv(all_predictions, csv_filename)
    print(f"预测结果已保存到: {csv_filename}")
    
    # 2. 保存TXT结果（包含时间统计）
    txt_filename = f'{save_dir}/test_results_{model_name}_{dataset_name}_{timestamp}.txt'
    save_results_to_txt(test_acc, confusion_matrix, class_names, txt_filename, 
                       dataset_name, csv_filename, total_time, avg_time_per_image, test_num)
    print(f"详细结果已保存到: {txt_filename}")
    
    return test_acc, confusion_matrix, csv_filename, txt_filename, total_time, avg_time_per_image

if __name__ == '__main__':
    # 记录整个脚本开始时间
    script_start_time = time.time()
    
    # 1. 选择数据集
    dataset_name = 'dogs77'  # 可选: 'oxford_pets', 'stanford_dogs', 'cats_vs_dogs', 'dogs77'
    
    # 2. 根据GPU显存调整batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16 if str(device) == 'cuda' else 8
    
    print(f"\n配置信息:")
    print(f"  数据集: {dataset_name}")
    print(f"  设备: {device}")
    print(f"  批量大小: {batch_size}")
    
    # 3. 获取数据加载器
    print("\n加载数据...")
    data_load_start = time.time()
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        augment=False
    )
    data_load_time = time.time() - data_load_start
    print(f"数据加载完成，耗时: {data_load_time:.2f}秒")
    
    # 4. 加载类别名称
    dataset_path = f'data/processed/{dataset_name}'
    class_names = load_class_names(dataset_path)
    
    if class_names is None or len(class_names) != num_classes:
        print(f"警告: 无法从数据集加载类别名称或数量不匹配")
        print(f"  期望类别数: {num_classes}")
        print(f"  使用默认类别名称...")
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    print(f"  类别名称: {class_names}")
    
    # 5. 创建模型
    model_create_start = time.time()
    model = VGG16(num_classes=num_classes)
    model_create_time = time.time() - model_create_start
    print(f"模型创建完成，耗时: {model_create_time:.2f}秒")
    
    # 6. 加载训练好的模型权重
    weight_load_start = time.time()
    model_weight_path = 'models/vgg/weights/vgg16_best_dogs77_20260426_025614.pth'
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path, weights_only=True))
        print(f"已加载模型权重: {model_weight_path}")
    else:
        weights_dir = 'models/vgg/weights'
        if os.path.exists(weights_dir):
            weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth') and 'vgg16' in f]
            if weight_files:
                weight_files.sort(reverse=True)
                latest_weight = os.path.join(weights_dir, weight_files[0])
                model.load_state_dict(torch.load(latest_weight, weights_only=True))
                print(f"已加载最新模型权重: {latest_weight}")
            else:
                print("错误: 未找到模型权重文件")
                exit(1)
        else:
            print("错误: 权重目录不存在")
            exit(1)
    weight_load_time = time.time() - weight_load_start
    print(f"权重加载完成，耗时: {weight_load_time:.2f}秒")
    
    # 7. 测试模型并保存结果
    print("\n开始测试...")
    test_acc, confusion_matrix, csv_path, txt_path, total_time, avg_time_per_image = test_model(
        model, test_loader, class_names, dataset_name)
    
    # 8. 计算总脚本运行时间
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    
    # 9. 输出最终结果
    print(f"\n{'='*60}")
    print(f"测试完成!")
    print(f"数据集: {dataset_name}")
    print(f"最终测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"时间统计:")
    print(f"  数据加载: {data_load_time:.2f}秒")
    print(f"  模型创建: {model_create_time:.2f}秒")
    print(f"  权重加载: {weight_load_time:.2f}秒")
    print(f"  推理时间: {format_time(total_time)}")
    print(f"  总脚本时间: {format_time(total_script_time)}")
    print(f"性能指标:")
    print(f"  平均每张图片: {avg_time_per_image*1000:.2f}毫秒")
    print(f"  处理速度: {len(test_loader.dataset)/total_time:.2f} 张/秒")
    print(f"生成文件:")
    print(f"  1. {os.path.basename(txt_path)} - 详细测试报告")
    print(f"  2. {os.path.basename(csv_path)} - 所有预测结果")
    print(f"文件位置: models/vgg/results/")
    print(f"{'='*60}\n")
    
    # 10. 显示简要混淆矩阵
    print("各类别准确率:")
    for i in range(len(class_names)):
        correct = confusion_matrix[i][i]
        total = sum(confusion_matrix[i])
        accuracy = correct / total if total > 0 else 0
        print(f"  {class_names[i]}: {accuracy:.2%} ({correct}/{total})")