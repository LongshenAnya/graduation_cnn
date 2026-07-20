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
# 2. 字号：图中文字清晰可辨，坐标轴标签 18pt，刻度 15pt
# 3. 坐标轴：必须标注"量、标准规定符号、单位"
# 4. 图例：必须包含，说明每条曲线的含义
# 5. 格式：导出为矢量图 .pdf 和 .svg（双格式保存）
# 6. 数据点：添加实心标记点，便于区分曲线
# ============================================================

def setup_chinese_font():
    """配置中文字体（宋体）和英文字体（Times New Roman）"""
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']  # SimSun 即宋体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['mathtext.fontset'] = 'stix'   # 数学公式使用 Times 风格


def matplot_acc_loss(train_process, save_plot=True, dataset_name='cats_vs_dogs'):
    """
    绘制训练曲线（符合重庆邮电大学本科毕业论文格式规范）
    
    模板依据：
    - 图应有"自明性"，只看图、图题和图注就能理解
    - 纵横坐标必须标注"量、标准规定符号、单位"
    - 必须有图例
    - 导出矢量图（.pdf 和 .svg 双格式）
    - 曲线添加数据点标记，便于区分
    
    参数:
        train_process: DataFrame，包含 epoch, train_loss_all, val_loss_all, 
                       train_acc_all, val_acc_all 列
        save_plot: bool，是否保存图片
        dataset_name: str，数据集名称，用于文件命名
    
    返回:
        plot_path: str，保存的图片路径（PDF格式）
    """
    
    # ===== 1. 配置字体（模板要求：中文宋体，英文 Times New Roman） =====
    setup_chinese_font()
    
    # ===== 2. 设置图形大小和样式 =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.25)  # 子图间距
    
    epochs = train_process['epoch'].values
    train_loss = train_process['train_loss_all'].values
    val_loss = train_process['val_loss_all'].values
    train_acc = train_process['train_acc_all'].values
    val_acc = train_process['val_acc_all'].values
    
    # ===== 3. 绘制损失曲线（左图） =====
    ax1 = axes[0]
    
    # 曲线样式：
    # - 训练损失：蓝色实线 + 实心圆形标记点（每2个epoch标一个，避免太密集）
    # - 验证损失：红色虚线 + 实心方形标记点
    line1, = ax1.plot(epochs, train_loss, 'b-', linewidth=1.5, 
                      marker='o', markersize=4, markevery=2,
                      label='训练损失 (Train Loss)')
    line2, = ax1.plot(epochs, val_loss, 'r--', linewidth=1.5, 
                      marker='s', markersize=4, markevery=2,
                      label='验证损失 (Val Loss)')
    
    # 坐标轴标签（模板要求：量、标准规定符号、单位）
    # 字体大小：原12pt × 1.5 = 18pt
    ax1.set_xlabel('训练轮次 Epoch', fontsize=18)
    ax1.set_ylabel('损失值 Loss', fontsize=18)
    
    # 图例
    # 字体大小：原10pt × 1.5 = 15pt
    ax1.legend(loc='upper right', fontsize=15, frameon=True, fancybox=False)
    
    # 网格（增强可读性）
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 设置坐标轴刻度字体大小
    # 字体大小：原10pt × 1.5 = 15pt
    ax1.tick_params(axis='both', labelsize=15)
    
    # ===== 4. 绘制准确率曲线（右图） =====
    ax2 = axes[1]
    
    # 曲线样式：
    # - 训练准确率：蓝色实线 + 实心圆形标记点
    # - 验证准确率：红色虚线 + 实心方形标记点
    line3, = ax2.plot(epochs, train_acc, 'b-', linewidth=1.5, 
                      marker='o', markersize=4, markevery=2,
                      label='训练准确率 (Train Acc)')
    line4, = ax2.plot(epochs, val_acc, 'r--', linewidth=1.5, 
                      marker='s', markersize=4, markevery=2,
                      label='验证准确率 (Val Acc)')
    
    # 坐标轴标签（模板要求：量、标准规定符号、单位）
    # 字体大小：原12pt × 1.5 = 18pt
    ax2.set_xlabel('训练轮次 Epoch', fontsize=18)
    ax2.set_ylabel('准确率 Accuracy', fontsize=18)
    
    # 格式化 y 轴为百分比
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 图例
    # 字体大小：原10pt × 1.5 = 15pt
    ax2.legend(loc='lower right', fontsize=15, frameon=True, fancybox=False)
    
    # 网格
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 设置坐标轴刻度字体大小
    # 字体大小：原10pt × 1.5 = 15pt
    ax2.tick_params(axis='both', labelsize=15)
    
    # ===== 5. 调整布局（防止边界裁剪） =====
    plt.subplots_adjust(left=0.08, right=0.95, wspace=0.28)
    plt.tight_layout(pad=0.5)
    
    # ===== 6. 保存图片（PDF 和 SVG 双矢量格式） =====
    plot_path = None
    if save_plot:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_dir = 'models/vgg/results'
        os.makedirs(plot_dir, exist_ok=True)
        
        # 同时保存 PDF 和 SVG 两种矢量格式
        plot_path_pdf = f'{plot_dir}/training_plot_{dataset_name}_{timestamp}.pdf'
        plot_path_svg = f'{plot_dir}/training_plot_{dataset_name}_{timestamp}.svg'
        plot_path_png = f'{plot_dir}/training_plot_{dataset_name}_{timestamp}.png'  # 预览用
        
        # 保存矢量图
        plt.savefig(plot_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_png, format='png', dpi=150, bbox_inches='tight')
        
        print(f"\n{'='*60}")
        print(f"训练曲线已保存（符合重庆邮电大学本科毕业论文格式规范）")
        print(f"矢量图（.pdf）: {plot_path_pdf}")
        print(f"矢量图（.svg）: {plot_path_svg}")
        print(f"预览图（.png）: {plot_path_png}")
        print(f"提示：推荐使用 .pdf 插入 Word，兼容性最佳")
        print(f"{'='*60}")
        
        plot_path = plot_path_pdf  # 返回 PDF 路径作为主文件
    
    # 显示图片
    plt.show()
    
    return plot_path


def matplot_acc_loss_separate(train_process, save_plot=True, dataset_name='cats_vs_dogs'):
    """
    分开绘制损失和准确率曲线（两张独立的图）
    
    适用场景：当需要将两张图分别插入论文不同位置时使用
    模板依据：每张图单独编号，如 图4.2、图4.3
    
    参数:
        train_process: DataFrame，包含训练数据
        save_plot: bool，是否保存图片
        dataset_name: str，数据集名称
    
    返回:
        loss_plot_path: str，损失曲线图路径（PDF格式）
        acc_plot_path: str，准确率曲线图路径（PDF格式）
    """
    
    # 配置字体
    setup_chinese_font()
    
    epochs = train_process['epoch'].values
    train_loss = train_process['train_loss_all'].values
    val_loss = train_process['val_loss_all'].values
    train_acc = train_process['train_acc_all'].values
    val_acc = train_process['val_acc_all'].values
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = 'models/vgg/results'
    os.makedirs(plot_dir, exist_ok=True)
    
    # ===== 图1：损失曲线 =====
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    # 添加数据点标记（实心）
    ax1.plot(epochs, train_loss, 'b-', linewidth=1.5, 
             marker='o', markersize=4, markevery=2,
             label='训练损失 (Train Loss)')
    ax1.plot(epochs, val_loss, 'r--', linewidth=1.5, 
             marker='s', markersize=4, markevery=2,
             label='验证损失 (Val Loss)')
    
    # 坐标轴标签
    # 字体大小：原12pt × 1.5 = 18pt
    ax1.set_xlabel('训练轮次 Epoch', fontsize=18)
    ax1.set_ylabel('损失值 Loss', fontsize=18)
    
    # 图例
    # 字体大小：原10pt × 1.5 = 15pt
    ax1.legend(loc='upper right', fontsize=15, frameon=True, fancybox=False)
    
    # 网格
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 刻度字体大小
    # 字体大小：原10pt × 1.5 = 15pt
    ax1.tick_params(axis='both', labelsize=15)
    
    plt.tight_layout(pad=0.5)
    
    # 保存双格式
    loss_plot_pdf = f'{plot_dir}/loss_curve_{dataset_name}_{timestamp}.pdf'
    loss_plot_svg = f'{plot_dir}/loss_curve_{dataset_name}_{timestamp}.svg'
    plt.savefig(loss_plot_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(loss_plot_svg, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(loss_plot_pdf.replace('.pdf', '.png'), format='png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig1)
    
    # ===== 图2：准确率曲线 =====
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    # 添加数据点标记（实心）
    ax2.plot(epochs, train_acc, 'b-', linewidth=1.5, 
             marker='o', markersize=4, markevery=2,
             label='训练准确率 (Train Acc)')
    ax2.plot(epochs, val_acc, 'r--', linewidth=1.5, 
             marker='s', markersize=4, markevery=2,
             label='验证准确率 (Val Acc)')
    
    # 坐标轴标签
    # 字体大小：原12pt × 1.5 = 18pt
    ax2.set_xlabel('训练轮次 Epoch', fontsize=18)
    ax2.set_ylabel('准确率 Accuracy', fontsize=18)
    
    # 格式化 y 轴为百分比
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 图例
    # 字体大小：原10pt × 1.5 = 15pt
    ax2.legend(loc='lower right', fontsize=15, frameon=True, fancybox=False)
    
    # 网格
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 刻度字体大小
    # 字体大小：原10pt × 1.5 = 15pt
    ax2.tick_params(axis='both', labelsize=15)
    
    plt.tight_layout(pad=0.5)
    
    # 保存双格式
    acc_plot_pdf = f'{plot_dir}/accuracy_curve_{dataset_name}_{timestamp}.pdf'
    acc_plot_svg = f'{plot_dir}/accuracy_curve_{dataset_name}_{timestamp}.svg'
    plt.savefig(acc_plot_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(acc_plot_svg, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(acc_plot_pdf.replace('.pdf', '.png'), format='png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig2)
    
    print(f"\n损失曲线已保存（PDF）: {loss_plot_pdf}")
    print(f"准确率曲线已保存（PDF）: {acc_plot_pdf}")
    print(f"提示：推荐使用 .pdf 插入 Word，兼容性最佳")
    
    return loss_plot_pdf, acc_plot_pdf


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    # 模拟数据（实际使用时从 train_model_process 返回的 train_process 获取）
    epochs = list(range(1, 48))
    train_loss = [0.6838, 0.6614, 0.6363, 0.6073, 0.5906, 0.5560, 0.5221, 0.4908, 0.4572, 0.4300,
                  0.3980, 0.3711, 0.3371, 0.3176, 0.2896, 0.2772, 0.2643, 0.2493, 0.2407, 0.2332,
                  0.2253, 0.2197, 0.2151, 0.2092, 0.2049, 0.2000, 0.1974, 0.1921, 0.1872, 0.1817,
                  0.1836, 0.1781, 0.1724, 0.1722, 0.1698, 0.1673, 0.1627, 0.1615, 0.1590, 0.1585,
                  0.1506, 0.1548, 0.1508, 0.1500, 0.1461, 0.1473, 0.1428]
    val_loss = [0.6664, 0.6246, 0.5818, 0.5726, 0.5144, 0.4885, 0.4896, 0.4291, 0.3856, 0.3808,
                0.3545, 0.3160, 0.2913, 0.2583, 0.2491, 0.2790, 0.2474, 0.2378, 0.2304, 0.2137,
                0.2151, 0.2228, 0.2087, 0.2118, 0.2055, 0.2058, 0.1898, 0.2012, 0.1939, 0.1952,
                0.1820, 0.1947, 0.1813, 0.1850, 0.1750, 0.1828, 0.1777, 0.1966, 0.1734, 0.1710,
                0.1695, 0.1627, 0.1655, 0.1666, 0.1709, 0.1699, 0.1766]
    
    # 模拟准确率数据
    train_acc = [0.5565, 0.6102, 0.6490, 0.6760, 0.6955, 0.7285, 0.7525, 0.7770, 0.8014, 0.8194,
                 0.8388, 0.8582, 0.8748, 0.8870, 0.9020, 0.9088, 0.9189, 0.9259, 0.9303, 0.9349,
                 0.9384, 0.9417, 0.9452, 0.9456, 0.9501, 0.9517, 0.9528, 0.9568, 0.9579, 0.9624,
                 0.9613, 0.9631, 0.9660, 0.9655, 0.9674, 0.9684, 0.9697, 0.9717, 0.9727, 0.9722,
                 0.9777, 0.9750, 0.9764, 0.9766, 0.9805, 0.9785, 0.9803]
    val_acc = [0.5998, 0.6591, 0.7077, 0.7205, 0.7614, 0.7761, 0.7809, 0.8271, 0.8533, 0.8485,
               0.8659, 0.8889, 0.9060, 0.9209, 0.9279, 0.9142, 0.9241, 0.9356, 0.9388, 0.9444,
               0.9463, 0.9391, 0.9476, 0.9487, 0.9519, 0.9495, 0.9546, 0.9522, 0.9535, 0.9548,
               0.9581, 0.9532, 0.9591, 0.9575, 0.9639, 0.9610, 0.9647, 0.9551, 0.9650, 0.9650,
               0.9663, 0.9685, 0.9669, 0.9679, 0.9666, 0.9666, 0.9637]
    
    train_process = pd.DataFrame({
        'epoch': epochs,
        'train_loss_all': train_loss,
        'val_loss_all': val_loss,
        'train_acc_all': train_acc,
        'val_acc_all': val_acc
    })
    
    # 绘制合并图（一张图包含两个子图）
    plot_path = matplot_acc_loss(train_process, save_plot=True, dataset_name='cats_vs_dogs')
    
    # 绘制分开的图（两张独立的图，适合分别插入论文）
    # loss_path, acc_path = matplot_acc_loss_separate(train_process, save_plot=True, dataset_name='cats_vs_dogs')