try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None
    transforms = None

from PIL import Image
import io

def preprocess_image(image_file):
    """
    预处理上传的图片文件
    - 缩放至224x224
    - 转换为Tensor
    - 归一化
    返回: tensor (shape: [1,3,224,224])
    """
    if torch is None or transforms is None:
        raise ImportError("PyTorch或torchvision未安装")
    
    # 打开图片
    image = Image.open(image_file).convert('RGB')
    
    # 定义预处理变换
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # 缩放至224x224
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])
    
    # 应用预处理
    tensor = preprocess(image)
    
    # 添加batch维度
    tensor = tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]
    
    return tensor