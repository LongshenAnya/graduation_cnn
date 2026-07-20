import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import datetime

# ============================================================
# 设置 Matplotlib 后端为 SVG（支持矢量图导出）
# ============================================================
matplotlib.use('SVG')

# ============================================================
# 字体配置（模板要求：中文宋体，英文 Times New Roman）
# ============================================================
def setup_chinese_font():
    """配置中文字体（宋体）和英文字体（Times New Roman）"""
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'


def plot_precision_bar(datasets, vgg16_data, resnet34_data, mobilenet_data, 
                       output_dir='figures/chapter6', save_plot=True):
    """
    绘制精确率对比柱状图
    
    参数:
        datasets: list，数据集名称列表
        vgg16_data: list，VGG16 在各数据集上的精确率
        resnet34_data: list，ResNet34 在各数据集上的精确率
        mobilenet_data: list，MobileNetV1 在各数据集上的精确率
        output_dir: str，输出目录
        save_plot: bool，是否保存图片
    """
    setup_chinese_font()
    
    # ===== 图形设置（高度从6增加到6.5） =====
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = ['#2E6EB5', '#D9643A', '#3A8C5C']
    
    # ===== 绘制柱状图 =====
    bars1 = ax.bar(x - width, vgg16_data, width, label='VGG16', 
                   color=colors[0], edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x, resnet34_data, width, label='ResNet34', 
                   color=colors[1], edgecolor='black', linewidth=0.6)
    bars3 = ax.bar(x + width, mobilenet_data, width, label='MobileNetV1', 
                   color=colors[2], edgecolor='black', linewidth=0.6)
    
    # ===== 在柱顶添加数值标签（去掉百分号） =====
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.2,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=18)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # ===== 坐标轴设置 =====
    ax.set_xlabel('数据集', fontsize=23, labelpad=12)
    ax.set_ylabel('精确率 (%)', fontsize=23, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylim(0, 112)
    
    # ===== 图例 =====
    ax.legend(loc='upper right', fontsize=18, frameon=True, fancybox=False)
    
    # ===== 网格 =====
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # ===== 保存图片 =====
    plot_path = None
    if save_plot:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path_pdf = f'{output_dir}/fig6.2_precision_comparison_{timestamp}.pdf'
        plot_path_svg = f'{output_dir}/fig6.2_precision_comparison_{timestamp}.svg'
        
        plt.savefig(plot_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', dpi=300, bbox_inches='tight')
        
        print(f"\n精确率柱状图已保存到: {output_dir}")
        print(f"  PDF: {os.path.basename(plot_path_pdf)}")
        print(f"  SVG: {os.path.basename(plot_path_svg)}")
        
        plot_path = plot_path_pdf
    
    plt.show()
    
    return plot_path


def plot_recall_bar(datasets, vgg16_data, resnet34_data, mobilenet_data,
                    output_dir='figures/chapter6', save_plot=True):
    """
    绘制召回率对比柱状图
    
    参数:
        datasets: list，数据集名称列表
        vgg16_data: list，VGG16 在各数据集上的召回率
        resnet34_data: list，ResNet34 在各数据集上的召回率
        mobilenet_data: list，MobileNetV1 在各数据集上的召回率
        output_dir: str，输出目录
        save_plot: bool，是否保存图片
    """
    setup_chinese_font()
    
    # ===== 图形设置（高度从6增加到6.5） =====
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = ['#2E6EB5', '#D9643A', '#3A8C5C']
    
    # ===== 绘制柱状图 =====
    bars1 = ax.bar(x - width, vgg16_data, width, label='VGG16', 
                   color=colors[0], edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x, resnet34_data, width, label='ResNet34', 
                   color=colors[1], edgecolor='black', linewidth=0.6)
    bars3 = ax.bar(x + width, mobilenet_data, width, label='MobileNetV1', 
                   color=colors[2], edgecolor='black', linewidth=0.6)
    
    # ===== 在柱顶添加数值标签（去掉百分号） =====
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.2,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=18)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # ===== 坐标轴设置 =====
    ax.set_xlabel('数据集', fontsize=23, labelpad=12)
    ax.set_ylabel('召回率 (%)', fontsize=23, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylim(0, 112)
    
    # ===== 图例 =====
    ax.legend(loc='upper right', fontsize=18, frameon=True, fancybox=False)
    
    # ===== 网格 =====
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # ===== 保存图片 =====
    plot_path = None
    if save_plot:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path_pdf = f'{output_dir}/fig6.3_recall_comparison_{timestamp}.pdf'
        plot_path_svg = f'{output_dir}/fig6.3_recall_comparison_{timestamp}.svg'
        
        plt.savefig(plot_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', dpi=300, bbox_inches='tight')
        
        print(f"\n召回率柱状图已保存到: {output_dir}")
        print(f"  PDF: {os.path.basename(plot_path_pdf)}")
        print(f"  SVG: {os.path.basename(plot_path_svg)}")
        
        plot_path = plot_path_pdf
    
    plt.show()
    
    return plot_path


def plot_f1_bar(datasets, vgg16_data, resnet34_data, mobilenet_data,
                output_dir='figures/chapter6', save_plot=True):
    """
    绘制F1分数对比柱状图
    
    参数:
        datasets: list，数据集名称列表
        vgg16_data: list，VGG16 在各数据集上的F1分数
        resnet34_data: list，ResNet34 在各数据集上的F1分数
        mobilenet_data: list，MobileNetV1 在各数据集上的F1分数
        output_dir: str，输出目录
        save_plot: bool，是否保存图片
    """
    setup_chinese_font()
    
    # ===== 图形设置（高度从6增加到6.5） =====
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = ['#2E6EB5', '#D9643A', '#3A8C5C']
    
    # ===== 绘制柱状图 =====
    bars1 = ax.bar(x - width, vgg16_data, width, label='VGG16', 
                   color=colors[0], edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x, resnet34_data, width, label='ResNet34', 
                   color=colors[1], edgecolor='black', linewidth=0.6)
    bars3 = ax.bar(x + width, mobilenet_data, width, label='MobileNetV1', 
                   color=colors[2], edgecolor='black', linewidth=0.6)
    
    # ===== 在柱顶添加数值标签（去掉百分号） =====
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.2,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=18)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # ===== 坐标轴设置 =====
    ax.set_xlabel('数据集', fontsize=23, labelpad=12)
    ax.set_ylabel('F1分数 (%)', fontsize=23, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylim(0, 112)
    
    # ===== 图例 =====
    ax.legend(loc='upper right', fontsize=18, frameon=True, fancybox=False)
    
    # ===== 网格 =====
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # ===== 保存图片 =====
    plot_path = None
    if save_plot:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path_pdf = f'{output_dir}/fig6.4_f1_comparison_{timestamp}.pdf'
        plot_path_svg = f'{output_dir}/fig6.4_f1_comparison_{timestamp}.svg'
        
        plt.savefig(plot_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', dpi=300, bbox_inches='tight')
        
        print(f"\nF1分数柱状图已保存到: {output_dir}")
        print(f"  PDF: {os.path.basename(plot_path_pdf)}")
        print(f"  SVG: {os.path.basename(plot_path_svg)}")
        
        plot_path = plot_path_pdf
    
    plt.show()
    
    return plot_path


def plot_all_metrics_together(datasets, precision_data, recall_data, f1_data,
                              output_dir='figures/chapter6', save_plot=True):
    """
    将精确率、召回率、F1分数绘制在一张图中（三个子图并排）
    
    参数:
        datasets: list，数据集名称列表
        precision_data: dict，包含 'VGG16', 'ResNet34', 'MobileNetV1' 三个键的精确率数据
        recall_data: dict，包含 'VGG16', 'ResNet34', 'MobileNetV1' 三个键的召回率数据
        f1_data: dict，包含 'VGG16', 'ResNet34', 'MobileNetV1' 三个键的F1分数数据
        output_dir: str，输出目录
        save_plot: bool，是否保存图片
    """
    setup_chinese_font()
    
    # ===== 创建三个子图 =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    x = np.arange(len(datasets))
    width = 0.25
    colors = ['#2E6EB5', '#D9643A', '#3A8C5C']
    
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=14)
    
    # ===== 子图1：精确率 =====
    ax1 = axes[0]
    bars1_1 = ax1.bar(x - width, precision_data['VGG16'], width, label='VGG16', color=colors[0], edgecolor='black', linewidth=0.5)
    bars1_2 = ax1.bar(x, precision_data['ResNet34'], width, label='ResNet34', color=colors[1], edgecolor='black', linewidth=0.5)
    bars1_3 = ax1.bar(x + width, precision_data['MobileNetV1'], width, label='MobileNetV1', color=colors[2], edgecolor='black', linewidth=0.5)
    add_value_labels(ax1, bars1_1)
    add_value_labels(ax1, bars1_2)
    add_value_labels(ax1, bars1_3)
    ax1.set_xlabel('数据集', fontsize=20)
    ax1.set_ylabel('精确率 (%)', fontsize=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=18, rotation=15)
    ax1.set_ylim(0, 112)
    ax1.legend(loc='upper right', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) 精确率对比', fontsize=22, pad=12)
    
    # ===== 子图2：召回率 =====
    ax2 = axes[1]
    bars2_1 = ax2.bar(x - width, recall_data['VGG16'], width, label='VGG16', color=colors[0], edgecolor='black', linewidth=0.5)
    bars2_2 = ax2.bar(x, recall_data['ResNet34'], width, label='ResNet34', color=colors[1], edgecolor='black', linewidth=0.5)
    bars2_3 = ax2.bar(x + width, recall_data['MobileNetV1'], width, label='MobileNetV1', color=colors[2], edgecolor='black', linewidth=0.5)
    add_value_labels(ax2, bars2_1)
    add_value_labels(ax2, bars2_2)
    add_value_labels(ax2, bars2_3)
    ax2.set_xlabel('数据集', fontsize=20)
    ax2.set_ylabel('召回率 (%)', fontsize=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=18, rotation=15)
    ax2.set_ylim(0, 112)
    ax2.legend(loc='upper right', fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(b) 召回率对比', fontsize=22, pad=12)
    
    # ===== 子图3：F1分数 =====
    ax3 = axes[2]
    bars3_1 = ax3.bar(x - width, f1_data['VGG16'], width, label='VGG16', color=colors[0], edgecolor='black', linewidth=0.5)
    bars3_2 = ax3.bar(x, f1_data['ResNet34'], width, label='ResNet34', color=colors[1], edgecolor='black', linewidth=0.5)
    bars3_3 = ax3.bar(x + width, f1_data['MobileNetV1'], width, label='MobileNetV1', color=colors[2], edgecolor='black', linewidth=0.5)
    add_value_labels(ax3, bars3_1)
    add_value_labels(ax3, bars3_2)
    add_value_labels(ax3, bars3_3)
    ax3.set_xlabel('数据集', fontsize=20)
    ax3.set_ylabel('F1分数 (%)', fontsize=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, fontsize=18, rotation=15)
    ax3.set_ylim(0, 112)
    ax3.legend(loc='upper right', fontsize=14)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title('(c) F1分数对比', fontsize=22, pad=12)
    
    plt.tight_layout()
    
    # ===== 保存图片 =====
    plot_path = None
    if save_plot:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path_pdf = f'{output_dir}/fig6.2-6.4_all_metrics_{timestamp}.pdf'
        plot_path_svg = f'{output_dir}/fig6.2-6.4_all_metrics_{timestamp}.svg'
        
        plt.savefig(plot_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', dpi=300, bbox_inches='tight')
        
        print(f"\n合并柱状图已保存到: {output_dir}")
        print(f"  PDF: {os.path.basename(plot_path_pdf)}")
        print(f"  SVG: {os.path.basename(plot_path_svg)}")
        
        plot_path = plot_path_pdf
    
    plt.show()
    
    return plot_path


# ============================================================
# 主函数：集中管理所有数据参数
# ============================================================
def main():
    """
    主函数：在此处统一修改所有数据和参数
    """
    
    # ===== 1. 数据集名称 =====
    datasets = ['Cats vs Dogs', 'Oxford Pets', '犬种分类']
    
    # ===== 2. 输出目录 =====
    output_dir = r'C:\Users\longj\Desktop\毕业设计\运行截图及相关图片'
    
    # ===== 3. 精确率数据 =====
    precision_vgg16 = [98.29, 73.24, 74.64]
    precision_resnet34 = [99.20, 78.22, 75.82]
    precision_mobilenet = [99.04, 72.46, 74.44]
    
    # ===== 4. 召回率数据 =====
    recall_vgg16 = [98.29, 72.66, 73.44]
    recall_resnet34 = [99.20, 77.42, 75.05]
    recall_mobilenet = [99.04, 71.45, 73.21]
    
    # ===== 5. F1分数数据 =====
    f1_vgg16 = [98.29, 72.53, 73.72]
    f1_resnet34 = [99.20, 77.51, 75.30]
    f1_mobilenet = [99.04, 71.14, 73.69]
    
    # ===== 6. 是否保存图片 =====
    save_plot = True
    
    # ===== 7. 选择要生成的图表 =====
    generate_precision = True      # 生成精确率对比图（图6.2）
    generate_recall = True         # 生成召回率对比图（图6.3）
    generate_f1 = True             # 生成F1分数对比图（图6.4）
    generate_combined = False      # 生成合并图（三个子图并排）
    
    # ============================================================
    # 以下代码无需修改，自动根据上述参数生成图表
    # ============================================================
    
    print("=" * 60)
    print("开始生成性能对比柱状图")
    print("=" * 60)
    print(f"\n数据集: {datasets}")
    print(f"输出目录: {output_dir}")
    print(f"\n精确率数据: VGG16={precision_vgg16}, ResNet34={precision_resnet34}, MobileNetV1={precision_mobilenet}")
    print(f"召回率数据: VGG16={recall_vgg16}, ResNet34={recall_resnet34}, MobileNetV1={recall_mobilenet}")
    print(f"F1分数数据: VGG16={f1_vgg16}, ResNet34={f1_resnet34}, MobileNetV1={f1_mobilenet}")
    
    # 生成精确率对比图
    if generate_precision:
        print("\n" + "=" * 60)
        print("生成精确率对比图（图6.2）")
        print("=" * 60)
        plot_precision_bar(
            datasets=datasets,
            vgg16_data=precision_vgg16,
            resnet34_data=precision_resnet34,
            mobilenet_data=precision_mobilenet,
            output_dir=output_dir,
            save_plot=save_plot
        )
    
    # 生成召回率对比图
    if generate_recall:
        print("\n" + "=" * 60)
        print("生成召回率对比图（图6.3）")
        print("=" * 60)
        plot_recall_bar(
            datasets=datasets,
            vgg16_data=recall_vgg16,
            resnet34_data=recall_resnet34,
            mobilenet_data=recall_mobilenet,
            output_dir=output_dir,
            save_plot=save_plot
        )
    
    # 生成F1分数对比图
    if generate_f1:
        print("\n" + "=" * 60)
        print("生成F1分数对比图（图6.4）")
        print("=" * 60)
        plot_f1_bar(
            datasets=datasets,
            vgg16_data=f1_vgg16,
            resnet34_data=f1_resnet34,
            mobilenet_data=f1_mobilenet,
            output_dir=output_dir,
            save_plot=save_plot
        )
    
    # 生成合并图
    if generate_combined:
        print("\n" + "=" * 60)
        print("生成合并对比图")
        print("=" * 60)
        
        precision_data = {
            'VGG16': precision_vgg16,
            'ResNet34': precision_resnet34,
            'MobileNetV1': precision_mobilenet
        }
        recall_data = {
            'VGG16': recall_vgg16,
            'ResNet34': recall_resnet34,
            'MobileNetV1': recall_mobilenet
        }
        f1_data = {
            'VGG16': f1_vgg16,
            'ResNet34': f1_resnet34,
            'MobileNetV1': f1_mobilenet
        }
        
        plot_all_metrics_together(
            datasets=datasets,
            precision_data=precision_data,
            recall_data=recall_data,
            f1_data=f1_data,
            output_dir=output_dir,
            save_plot=save_plot
        )
    
    print("\n" + "=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)


# ============================================================
# 程序入口
# ============================================================
if __name__ == '__main__':
    main()