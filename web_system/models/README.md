# 模型文件说明
# 请将训练好的PyTorch模型文件放置在此目录下：
# - resnet_catdog.pth
# - vgg_catdog.pth
# - mobilenet_catdog.pth
#
# 模型要求：
# - 输入尺寸：224x224
# - 输出：2类（猫/狗）
# - 类别对应：0=猫, 1=狗
# - 使用torch.save()保存的完整模型