"""
VGG16模型训练脚本
针对宠物图像分类任务优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from model import VGG16
from dataloader import get_dataloaders

def train_model(model, train_loader, val_loader, num_epochs=20, device=None):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        device: 训练设备（GPU/CPU）
    
    Returns:
        train_process: 训练过程记录
        best_model_state: 最佳模型状态
    """
    
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加L2正则化
    
    # 学习率调度器（每10个epoch学习率乘以0.1）
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 记录训练过程
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_acc = 0.0
    best_model_state = None
    
    start_time = time.time()
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # 每50个batch打印一次进度
            if batch_idx % 50 == 0:
                batch_acc = torch.sum(preds == labels.data).float() / inputs.size(0)
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")
        
        # 计算训练集指标
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_corrects.double() / total_samples
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_samples += inputs.size(0)
        
        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_corrects.double() / val_samples
        
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        
        # 打印当前epoch结果
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        print(f"Learning Rate: {history['lr'][-1]:.6f}")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"✅ 保存最佳模型，验证准确率: {best_acc:.4f}")
    
    # 训练完成
    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成! 总耗时: {training_time//60:.0f}分 {training_time%60:.0f}秒")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"{'='*60}")
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_state)
    
    # 转换为DataFrame
    train_process = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'lr': history['lr']
    })
    
    return train_process, best_model_state

def save_model(model, model_state, save_dir='weights', model_name='vgg16'):
    """保存模型和训练状态"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 保存模型权重
    torch.save(model_state, save_path / f'{model_name}_best.pth')
    
    # 保存模型架构和权重
    torch.save(model, save_path / f'{model_name}_full.pth')
    
    print(f"模型已保存到: {save_path}")
    print(f"  - {model_name}_best.pth: 仅权重")
    print(f"  - {model_name}_full.pth: 完整模型")

def plot_training_history(train_process):
    """绘制训练历史图表"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 损失曲线
    axes[0].plot(train_process['epoch'], train_process['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(train_process['epoch'], train_process['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(train_process['epoch'], train_process['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(train_process['epoch'], train_process['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # 学习率曲线
    axes[2].plot(train_process['epoch'], train_process['lr'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = Path('results')
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("开始训练VGG16模型...")
    
    # 1. 选择数据集（先从最简单的开始）
    dataset_name = 'cats_vs_dogs'  # 2分类，数据量大，训练快
    
    # 2. 根据GPU显存调整batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16 if str(device) == 'cuda' else 8
    
    print(f"\n配置信息:")
    print(f"  数据集: {dataset_name}")
    print(f"  设备: {device}")
    print(f"  批量大小: {batch_size}")
    
    # 3. 获取数据加载器
    print("\n加载数据...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        augment=False  # 第一阶段不使用数据增强
    )
    
    # 4. 创建模型
    print(f"\n创建VGG16模型 (类别数: {num_classes})...")
    model = VGG16(num_classes=num_classes)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 5. 训练模型
    print(f"\n开始训练...")
    train_process, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=80,
        device=device
    )
    
    # 6. 保存模型
    print(f"\n保存模型...")
    save_model(model, best_state, model_name=f'vgg16_{dataset_name}')
    
    # 7. 绘制训练曲线
    print(f"\n生成训练图表...")
    plot_training_history(train_process)
    
    # 8. 保存训练记录
    save_path = Path('results')
    train_process.to_csv(save_path / f'training_history_{dataset_name}.csv', index=False)
    print(f"训练记录已保存到: {save_path / 'training_history.csv'}")
    
    print(f"\n✅ VGG16在{dataset_name}数据集上的训练完成！")

if __name__ == '__main__':
    main()