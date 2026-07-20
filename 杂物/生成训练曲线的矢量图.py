import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import datetime
import numpy as np

# ============================================================
# 设置 Matplotlib 后端为 SVG（支持矢量图导出）
# ============================================================
matplotlib.use('SVG')

# ============================================================
# 绘图函数（符合重庆邮电大学本科毕业设计（论文）格式规范）
# ============================================================
# 模板依据：
# 1. 字体：中文宋体，英文 Times New Roman
# 2. 字号：坐标轴标签 18pt，刻度 15pt，图例 15pt
# 3. 坐标轴：必须标注"量、标准规定符号、单位"
# 4. 图例：必须包含，说明每条曲线的含义
# 5. 格式：导出为矢量图 .pdf 和 .svg（双格式保存）
# 6. 数据点：添加实心标记点，便于区分曲线
# ============================================================

def setup_chinese_font():
    """配置中文字体（宋体）和英文字体（Times New Roman）"""
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'


def plot_from_csv(csv_path, save_plot=True, dataset_name=None, max_epochs=None, output_dir=None):
    """
    从 CSV 文件读取训练数据并绘制曲线
    
    参数:
        csv_path: str，CSV 文件路径
        save_plot: bool，是否保存图片
        dataset_name: str，数据集名称，用于文件命名（默认从 CSV 文件名提取）
        max_epochs: int，显示的最大 epoch 数（None 表示全部显示）
        output_dir: str，图片保存目录（默认 'models/vgg/results'）
    
    返回:
        plot_path: str，保存的图片路径（PDF格式）
    """
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 检查必要的列是否存在
    required_columns = ['epoch', 'train_loss_all', 'val_loss_all', 'train_acc_all', 'val_acc_all']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV 文件缺少必要列: {col}")
    
    # 限制显示范围
    if max_epochs is not None:
        df = df[df['epoch'] <= max_epochs]
    
    # 提取数据
    epochs = df['epoch'].values
    train_loss = df['train_loss_all'].values
    val_loss = df['val_loss_all'].values
    train_acc = df['train_acc_all'].values
    val_acc = df['val_acc_all'].values
    
    # 自动提取数据集名称
    if dataset_name is None:
        import re
        basename = os.path.basename(csv_path)
        match = re.search(r'training_(.+?)_metrics', basename)
        if match:
            dataset_name = match.group(1)
        else:
            dataset_name = 'dataset'
    
    print(f"\n{'='*60}")
    print(f"读取 CSV 文件: {csv_path}")
    print(f"数据集: {dataset_name}")
    print(f"总 epoch 数: {len(df)}")
    print(f"显示 epoch 范围: {epochs[0]} - {epochs[-1]}")
    print(f"最佳验证准确率: {val_acc.max():.4f} (Epoch {epochs[np.argmax(val_acc)]})")
    print(f"{'='*60}")
    
    # 调用绘图函数
    train_process = df.copy()
    return matplot_acc_loss(train_process, save_plot=save_plot, 
                           dataset_name=dataset_name, output_dir=output_dir)


def matplot_acc_loss(train_process, save_plot=True, dataset_name='dataset', output_dir=None):
    """
    绘制训练曲线（符合重庆邮电大学本科毕业论文格式规范）
    
    参数:
        train_process: DataFrame，包含 epoch, train_loss_all, val_loss_all, 
                       train_acc_all, val_acc_all 列
        save_plot: bool，是否保存图片
        dataset_name: str，数据集名称，用于文件命名
        output_dir: str，图片保存目录（默认 'models/vgg/results'）
    
    返回:
        plot_path: str，保存的图片路径（PDF格式）
    """
    
    # ===== 1. 配置字体 =====
    setup_chinese_font()
    
    # ===== 2. 设置图形大小和样式 =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.25)
    
    epochs = train_process['epoch'].values
    train_loss = train_process['train_loss_all'].values
    val_loss = train_process['val_loss_all'].values
    train_acc = train_process['train_acc_all'].values
    val_acc = train_process['val_acc_all'].values
    
    # 计算标记点间隔（根据 epoch 总数自动调整）
    total_epochs = len(epochs)
    if total_epochs <= 50:
        markevery = 2
    elif total_epochs <= 100:
        markevery = 5
    else:
        markevery = 10
    
    # ===== 3. 绘制损失曲线（左图） =====
    ax1 = axes[0]
    
    ax1.plot(epochs, train_loss, 'b-', linewidth=1.5, 
             marker='o', markersize=4, markevery=markevery,
             label='训练损失 (Train Loss)')
    ax1.plot(epochs, val_loss, 'r--', linewidth=1.5, 
             marker='s', markersize=4, markevery=markevery,
             label='验证损失 (Val Loss)')
    
    # 坐标轴标签（字体大小：原12pt × 1.5 = 18pt）
    ax1.set_xlabel('训练轮次 Epoch', fontsize=18)
    ax1.set_ylabel('损失值 Loss', fontsize=18)
    
    # 图例（字体大小：原10pt × 1.5 = 15pt）
    ax1.legend(loc='upper right', fontsize=15, frameon=True, fancybox=False)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 刻度标签（字体大小：原10pt × 1.5 = 15pt）
    ax1.tick_params(axis='both', labelsize=15)
    
    # ===== 4. 绘制准确率曲线（右图） =====
    ax2 = axes[1]
    
    ax2.plot(epochs, train_acc, 'b-', linewidth=1.5, 
             marker='o', markersize=4, markevery=markevery,
             label='训练准确率 (Train Acc)')
    ax2.plot(epochs, val_acc, 'r--', linewidth=1.5, 
             marker='s', markersize=4, markevery=markevery,
             label='验证准确率 (Val Acc)')
    
    # 坐标轴标签（字体大小：原12pt × 1.5 = 18pt）
    ax2.set_xlabel('训练轮次 Epoch', fontsize=18)
    ax2.set_ylabel('准确率 Accuracy', fontsize=18)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 图例（字体大小：原10pt × 1.5 = 15pt）
    ax2.legend(loc='lower right', fontsize=15, frameon=True, fancybox=False)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 刻度标签（字体大小：原10pt × 1.5 = 15pt）
    ax2.tick_params(axis='both', labelsize=15)
    
    # ===== 5. 调整布局 =====
    plt.subplots_adjust(left=0.08, right=0.95, wspace=0.28)
    plt.tight_layout(pad=0.5)
    
    # ===== 6. 保存图片 =====
    plot_path = None
    if save_plot:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置保存目录
        if output_dir is None:
            plot_dir = 'models/vgg/results'
        else:
            plot_dir = output_dir
        
        os.makedirs(plot_dir, exist_ok=True)
        
        # 文件名包含 epoch 范围
        epoch_range = f"{epochs[0]}-{epochs[-1]}"
        plot_path_pdf = f'{plot_dir}/training_plot_{dataset_name}_epoch{epoch_range}_{timestamp}.pdf'
        plot_path_svg = f'{plot_dir}/training_plot_{dataset_name}_epoch{epoch_range}_{timestamp}.svg'
        plot_path_png = f'{plot_dir}/training_plot_{dataset_name}_epoch{epoch_range}_{timestamp}.png'
        
        plt.savefig(plot_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_png, format='png', dpi=150, bbox_inches='tight')
        
        print(f"\n{'='*60}")
        print(f"训练曲线已保存到: {plot_dir}")
        print(f"矢量图（.pdf）: {os.path.basename(plot_path_pdf)}")
        print(f"矢量图（.svg）: {os.path.basename(plot_path_svg)}")
        print(f"预览图（.png）: {os.path.basename(plot_path_png)}")
        print(f"提示：推荐使用 .pdf 插入 Word，兼容性最佳")
        print(f"{'='*60}")
        
        plot_path = plot_path_pdf
    
    plt.show()
    
    return plot_path


def plot_from_csv_separate(csv_path, save_plot=True, dataset_name=None, max_epochs=None, output_dir=None):
    """
    从 CSV 文件读取数据并分开绘制损失和准确率曲线（两张独立的图）
    
    参数:
        csv_path: str，CSV 文件路径
        save_plot: bool，是否保存图片
        dataset_name: str，数据集名称（默认从 CSV 文件名提取）
        max_epochs: int，显示的最大 epoch 数
        output_dir: str，图片保存目录（默认 'models/vgg/results'）
    
    返回:
        loss_path: str，损失曲线图路径
        acc_path: str，准确率曲线图路径
    """
    
    df = pd.read_csv(csv_path)
    
    if max_epochs is not None:
        df = df[df['epoch'] <= max_epochs]
    
    if dataset_name is None:
        import re
        basename = os.path.basename(csv_path)
        match = re.search(r'training_(.+?)_metrics', basename)
        dataset_name = match.group(1) if match else 'dataset'
    
    epochs = df['epoch'].values
    train_loss = df['train_loss_all'].values
    val_loss = df['val_loss_all'].values
    train_acc = df['train_acc_all'].values
    val_acc = df['val_acc_all'].values
    
    total_epochs = len(epochs)
    if total_epochs <= 50:
        markevery = 2
    elif total_epochs <= 100:
        markevery = 5
    else:
        markevery = 10
    
    setup_chinese_font()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_range = f"{epochs[0]}-{epochs[-1]}"
    
    # 设置保存目录
    if output_dir is None:
        plot_dir = 'models/vgg/results'
    else:
        plot_dir = output_dir
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # ===== 图1：损失曲线 =====
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.plot(epochs, train_loss, 'b-', linewidth=1.5, 
             marker='o', markersize=4, markevery=markevery,
             label='训练损失 (Train Loss)')
    ax1.plot(epochs, val_loss, 'r--', linewidth=1.5, 
             marker='s', markersize=4, markevery=markevery,
             label='验证损失 (Val Loss)')
    
    # 坐标轴标签（字体大小：原12pt × 1.5 = 18pt）
    ax1.set_xlabel('训练轮次 Epoch', fontsize=18)
    ax1.set_ylabel('损失值 Loss', fontsize=18)
    
    # 图例（字体大小：原10pt × 1.5 = 15pt）
    ax1.legend(loc='upper right', fontsize=15, frameon=True, fancybox=False)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 刻度标签（字体大小：原10pt × 1.5 = 15pt）
    ax1.tick_params(axis='both', labelsize=15)
    
    plt.tight_layout(pad=0.5)
    
    loss_plot_pdf = f'{plot_dir}/loss_curve_{dataset_name}_epoch{epoch_range}_{timestamp}.pdf'
    loss_plot_svg = f'{plot_dir}/loss_curve_{dataset_name}_epoch{epoch_range}_{timestamp}.svg'
    plt.savefig(loss_plot_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(loss_plot_svg, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(loss_plot_pdf.replace('.pdf', '.png'), format='png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig1)
    
    # ===== 图2：准确率曲线 =====
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    ax2.plot(epochs, train_acc, 'b-', linewidth=1.5, 
             marker='o', markersize=4, markevery=markevery,
             label='训练准确率 (Train Acc)')
    ax2.plot(epochs, val_acc, 'r--', linewidth=1.5, 
             marker='s', markersize=4, markevery=markevery,
             label='验证准确率 (Val Acc)')
    
    # 坐标轴标签（字体大小：原12pt × 1.5 = 18pt）
    ax2.set_xlabel('训练轮次 Epoch', fontsize=18)
    ax2.set_ylabel('准确率 Accuracy', fontsize=18)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 图例（字体大小：原10pt × 1.5 = 15pt）
    ax2.legend(loc='lower right', fontsize=15, frameon=True, fancybox=False)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 刻度标签（字体大小：原10pt × 1.5 = 15pt）
    ax2.tick_params(axis='both', labelsize=15)
    
    plt.tight_layout(pad=0.5)
    
    acc_plot_pdf = f'{plot_dir}/accuracy_curve_{dataset_name}_epoch{epoch_range}_{timestamp}.pdf'
    acc_plot_svg = f'{plot_dir}/accuracy_curve_{dataset_name}_epoch{epoch_range}_{timestamp}.svg'
    plt.savefig(acc_plot_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(acc_plot_svg, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(acc_plot_pdf.replace('.pdf', '.png'), format='png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig2)
    
    print(f"\n损失曲线已保存到: {plot_dir}")
    print(f"  PDF: {os.path.basename(loss_plot_pdf)}")
    print(f"准确率曲线已保存到: {plot_dir}")
    print(f"  PDF: {os.path.basename(acc_plot_pdf)}")
    
    return loss_plot_pdf, acc_plot_pdf


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    # CSV 文件路径（请修改为你的实际路径）
    csv_path = r"D:\study\graduation_cnn\models\mobilenet\备份\dogs77\有数据增强\training_20260417_175043_metrics.csv"
    
    # ===== 示例1：保存到默认路径 =====
    # plot_path = plot_from_csv(csv_path, max_epochs=100)
    
    # ===== 示例2：保存到自定义路径 =====
    plot_path = plot_from_csv(
        csv_path=csv_path,
        output_dir=r'D:\study\graduation_cnn\models\mobilenet\备份\dogs77\有数据增强',  # 自定义保存目录
        max_epochs=None,  # 显示全部 epoch
        save_plot=True
    )
    
    # ===== 示例3：绘制分开的图并保存到自定义路径 =====
    # loss_path, acc_path = plot_from_csv_separate(
    #     csv_path=csv_path,
    #     output_dir='figures/chapter4',
    #     max_epochs=100,
    #     save_plot=True
    # )
    
    # ===== 示例4：手动指定数据集名称 =====
    # plot_path = plot_from_csv(
    #     csv_path=csv_path,
    #     dataset_name='Stanford_Dogs',
    #     output_dir='figures/chapter4',
    #     max_epochs=100
    # )
    
    print(f"\n绘图完成！图片保存路径: {plot_path}")