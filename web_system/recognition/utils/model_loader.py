try:
    import torch
    import torch.nn as nn
    from torchvision import models
except ImportError:
    torch = None
    nn = None
    models = None

import os
from django.conf import settings

# 导入你的自定义模型
from models.resnet import Residual, ResNet34  # 注意导入 Residual
from models.vgg import VGG16
from models.mobilenet import MobileNetV1

# 导入品种映射工具
from .breed_mapper import breed_mapper

# 模型缓存 - 按任务分别缓存
_model_cache = {}


def get_device():
    """自动检测可用设备：优先使用CUDA，否则退回CPU。"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_name, task='catdog'):
    """
    按需加载模型

    Args:
        model_name: 模型名称 ('resnet', 'vgg', 'mobilenet')
        task: 任务类型 ('catdog' 或 'breeds')

    Returns:
        PyTorch模型实例
    """
    if torch is None:
        raise ImportError("PyTorch未安装")

    global _model_cache

    # 缓存键包含任务类型
    cache_key = f"{model_name}_{task}"

    # 如果模型已缓存，直接返回
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # 根据任务确定模型文件路径和类别数
    if task == 'catdog':
        model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}_catdog.pth')
        num_classes = 2
    elif task == 'breeds':
        model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}_dogbreed.pth')
        num_classes = breed_mapper.get_total_breeds()  # 76
    else:
        raise ValueError(f"未知的任务类型: {task}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 根据模型名称创建对应的模型实例
    if model_name == 'resnet':
        # 使用你的自定义ResNet34
        model = ResNet34(Residual=Residual, num_classes=num_classes)
    elif model_name == 'vgg':
        # 使用你的自定义VGG16
        model = VGG16(num_classes=num_classes)
    elif model_name == 'mobilenet':
        # 使用你的自定义MobileNetV1
        model = MobileNetV1(num_classes=num_classes, width_multiplier=1.0)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    # 自动选择设备：如果当前系统有可用GPU，则使用CUDA，否则使用CPU
    device = get_device()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 将模型移动到目标设备，并设置为评估模式
    model.to(device)
    model.eval()

    # 缓存模型
    _model_cache[cache_key] = model

    return model