import time
import torch
import torch.nn as nn
import pandas as pd
import copy
import os
import sys
import random  # 新增：用于验证采样
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from torch.amp import autocast, GradScaler  # 新增：混合精度

from model import MobileNetV1
from dataloader import get_dataloaders

# ===== 环境变量优化 =====
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ===== TF32加速 =====
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 定义一个简单的日志类，将输出同时写入控制台和文件
class TeeLogger:
    """同时输出到控制台和文件的简单日志类"""
    
    def __init__(self, log_file=None):
        # 默认保存到 models/mobilenet/results/training_{timestamp}.log
        if log_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = 'models/mobilenet/results'
            os.makedirs(log_dir, exist_ok=True)
            log_file = f'{log_dir}/training_{timestamp}.log'
        
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 立即写入文件
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        return False
        
    def close(self):
        if self.log:
            self.log.close()

def train_model_process(model, train_loader, val_loader, num_epochs=10, dataset_name=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ===== 梯度累积设置 =====
    accumulation_steps = 4  # 累积4步更新一次
    print(f"Gradient accumulation: {accumulation_steps} steps")
    print(f"Effective batch size: {train_loader.batch_size} × {accumulation_steps} = {train_loader.batch_size * accumulation_steps}")

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 优化器（学习率调整）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * (accumulation_steps ** 0.5))

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=1e-6
    )

    # 混合精度
    scaler = GradScaler('cuda')

    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    best_acc = 0.0
    patience = 15  # 早停耐心值
    patience_counter = 0  # 计数器
    best_epoch = 0  # 记录最佳模型所在的epoch
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()

    # 确保权重目录存在
    os.makedirs('models/mobilenet/weights', exist_ok=True)

    # ===== 验证比例设置 =====
    val_sample_ratio = 0.1  # 每次随机使用10%的验证数据
    total_val_batches = len(val_loader)
    val_sample_size = max(1, int(total_val_batches * val_sample_ratio))
    print(f"Validation: using {val_sample_size}/{total_val_batches} batches (10%) each epoch")

    # 训练循环
    for epoch in range(num_epochs):
        # 打印epoch信息
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # 初始化统计
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # ===== 训练阶段（带梯度累积） =====
        train_iterator = enumerate(train_loader)
        total_batches = len(train_loader)

        # 初始化梯度
        optimizer.zero_grad()
        
        for step, (inputs, labels) in train_iterator:
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 混合精度前向传播
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                pre_lab = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)
                # 关键：将损失除以累积步数
                loss = loss / accumulation_steps

            # 反向传播（累积梯度）
            scaler.scale(loss).backward()

            # 每 accumulation_steps 步更新一次参数
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 统计（注意：loss要乘回accumulation_steps才是真实值）
            train_loss += loss.item() * inputs.size(0) * accumulation_steps
            train_corrects += torch.sum(pre_lab == labels.data)
            train_num += inputs.size(0)

            # 每100个batch显示一次进度
            if (step + 1) % 100 == 0 or (step + 1) == total_batches:
                batch_acc = torch.sum(pre_lab == labels.data).item() / inputs.size(0)
                print(f"  Batch {step+1}/{total_batches} - Loss: {loss.item() * accumulation_steps:.4f}, Acc: {batch_acc:.4f}")

        # 处理最后可能剩余的梯度
        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ===== 验证阶段（随机采样10%） =====
        print(f"  Validating (random 10% sample)...")

        # 随机选择10%的batch索引
        val_batch_indices = random.sample(range(total_val_batches), val_sample_size)
        val_batch_indices.sort()
        
        with torch.no_grad():
            val_iterator = enumerate(val_loader)
            
            for step, (inputs, labels) in val_iterator:
                if step not in val_batch_indices:
                    continue
                    
                model.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                pre_lab = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(pre_lab == labels.data)
                val_num += inputs.size(0)

                # 每50个batch显示一次进度
                if (step + 1) % 200 == 0:
                    batch_acc = torch.sum(pre_lab == labels.data).item() / inputs.size(0)
                    print(f"    Val Batch {step+1}/{total_val_batches} - Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")

        # 计算epoch统计
        train_epoch_loss = train_loss / train_num
        train_epoch_acc = train_corrects.double().item() / train_num
        val_epoch_loss = val_loss / val_num if val_num > 0 else 0
        val_epoch_acc = val_corrects.double().item() / val_num if val_num > 0 else 0

        # 保存结果
        train_loss_all.append(train_epoch_loss)
        train_acc_all.append(train_epoch_acc)
        val_loss_all.append(val_epoch_loss)
        val_acc_all.append(val_epoch_acc)

        # 打印epoch结果
        print(f"\n  Train Loss: {train_epoch_loss:.4f} | Acc: {train_epoch_acc:.4f}")
        print(f"  Val Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f}")

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current learning rate: {current_lr:.6f}")

        # 保存最佳模型 + 早停
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch + 1
            print(f"  New best model saved! (Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs")

        # 早停判断
        if patience_counter >= patience:
            print(f"\n{'='*50}")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best validation accuracy: {best_acc:.4f} (achieved at epoch {best_epoch})")
            print(f"{'='*50}\n")
            break

        # 计算时间
        time_use = time.time() - since
        print(f"  Time elapsed: {time_use // 60:.0f}m {time_use % 60:.0f}s")

    # 保存最终最佳模型
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(best_model_wts, f'models/mobilenet/weights/mobilenetv1_best_{dataset_name}_{timestamp}.pth')
    print(f"\n{'='*50}")
    print(f"Training completed! Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: models/mobilenet/weights/mobilenetv1_best_{dataset_name}_{timestamp}.pth")
    print(f"{'='*50}\n")

    # 创建训练过程记录
    actual_epochs = len(train_loss_all)
    train_process = pd.DataFrame({
        "epoch": range(1, actual_epochs + 1),
        "train_loss_all": train_loss_all,
        "train_acc_all": train_acc_all,
        "val_loss_all": val_loss_all,
        "val_acc_all": val_acc_all
    })
    
    return train_process

# 绘制训练过程中的损失和准确率曲线
def matplot_acc_loss(train_process, save_plot=True):
    """绘制训练曲线，可选保存"""
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'ro-', 
             label='Train Loss', markersize=4, linewidth=1)
    plt.plot(train_process['epoch'], train_process.val_loss_all, 'bs-', 
             label='Val Loss', markersize=4, linewidth=1)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'ro-', 
             label='Train Acc', markersize=4, linewidth=1)
    plt.plot(train_process['epoch'], train_process.val_acc_all, 'bs-', 
             label='Val Acc', markersize=4, linewidth=1)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 自动保存图片
    if save_plot:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_dir = 'models/mobilenet/results'
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = f'{plot_dir}/training_plot_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to: {plot_path}")
    
    plt.show()
    
    return plot_path if save_plot else None

if __name__ == '__main__':
    from torch.multiprocessing import freeze_support
    freeze_support()
    # ===== 启用日志记录 =====
    # 创建 results 目录
    os.makedirs('models/mobilenet/results', exist_ok=True)
    
    # 开始记录日志
    logger = TeeLogger()
    sys.stdout = logger
    
    try:
        # 选择数据集
        dataset_name = 'dogs77'  # 或 'oxford_pets', 'stanford_dogs', 'cats_vs_dogs'

        # 获取数据加载器
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*60}")
        
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(
            dataset_name=dataset_name,
            batch_size=32,
            augment=True  # 先无增强
        )

        # 模型实例化
        print(f"Number of classes: {num_classes}")
        mobileNetV1 = MobileNetV1(num_classes=num_classes, width_multiplier=1.0)
        
        # 显示模型信息
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("Using device: CPU")

        # 训练模型
        print(f"\nStarting training for {60} epochs...")
        train_process = train_model_process(mobileNetV1, train_loader, val_loader, num_epochs=60, dataset_name=dataset_name)

        # 绘制并保存训练曲线
        print(f"\nGenerating training curves...")
        plot_path = matplot_acc_loss(train_process, save_plot=True)
        
        # 保存训练数据到CSV
        csv_path = logger.log_file.replace('.log', '_metrics.csv')
        train_process.to_csv(csv_path, index=False)
        print(f"Training metrics saved to: {csv_path}")
        
    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout = logger.terminal
        logger.close()
        
        print(f"\n{'='*60}")
        print(f"All output has been saved to: {logger.log_file}")
        print(f"{'='*60}")