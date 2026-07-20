try:
    import torch
except ImportError:
    torch = None

import time

def predict(image_tensor, model, class_names=None):
    """
    执行推理

    Args:
        image_tensor: 输入图像张量 (shape: [1,3,224,224])
        model: PyTorch模型
        class_names: 类别名称列表，如果为None则使用默认的猫狗分类

    Returns:
        tuple: (predicted_class, confidence, inference_time)
    """
    if torch is None:
        raise ImportError("PyTorch未安装")

    # 默认类别名称（猫狗分类）
    if class_names is None:
        class_names = ['cat', 'dog']

    # 确保输入张量与模型在同一设备上，避免CPU/GPU设备不匹配错误
    model_device = next(model.parameters()).device
    image_tensor = image_tensor.to(model_device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model(image_tensor)  # 得到logits

        # 使用softmax转成概率
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # 获取置信度和预测类别索引
        confidence, predicted = torch.max(probs, 1)

        # 根据类别名称列表获取预测结果
        predicted_index = predicted.item()
        if 0 <= predicted_index < len(class_names):
            predicted_class = class_names[predicted_index]
        else:
            predicted_class = f"unknown_{predicted_index}"

        confidence_value = confidence.item()

    inference_time = time.time() - start_time

    return predicted_class, confidence_value, inference_time