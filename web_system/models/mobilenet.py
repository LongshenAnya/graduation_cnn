import torch
from torch import nn
from torchsummary import summary


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 深度卷积 (Depthwise Convolution)
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                     stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 逐点卷积 (Pointwise Convolution)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=1.0):
        """
        MobileNet V1 实现
        
        参数:
            num_classes: 分类数量
            width_multiplier: 宽度乘数，用于控制模型大小 (α)
                            1.0: 标准MobileNet
                            0.75: 减小25%
                            0.5: 减小50%
        """
        super(MobileNetV1, self).__init__()
        
        def conv_bn(in_channels, out_channels, stride):
            """标准卷积块"""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # 计算经过宽度乘数调整的通道数
        def adjust_channels(channels):
            return int(channels * width_multiplier)
        
        # 根据论文的MobileNet V1结构
        # 输入: 3×224×224
        
        # 第一层: 标准卷积
        self.features = nn.Sequential(
            # 输入: 3×224×224
            conv_bn(3, adjust_channels(32), stride=2),  # 32×112×112
            
            # 深度可分离卷积块1
            DepthwiseSeparableConv(adjust_channels(32), adjust_channels(64), stride=1),  # 64×112×112
            
            # 深度可分离卷积块2
            DepthwiseSeparableConv(adjust_channels(64), adjust_channels(128), stride=2),  # 128×56×56
            DepthwiseSeparableConv(adjust_channels(128), adjust_channels(128), stride=1),  # 128×56×56
            
            # 深度可分离卷积块3
            DepthwiseSeparableConv(adjust_channels(128), adjust_channels(256), stride=2),  # 256×28×28
            DepthwiseSeparableConv(adjust_channels(256), adjust_channels(256), stride=1),  # 256×28×28
            
            # 深度可分离卷积块4
            DepthwiseSeparableConv(adjust_channels(256), adjust_channels(512), stride=2),  # 512×14×14
            
            # 5个连续的深度可分离卷积块 (论文中是5个)
            DepthwiseSeparableConv(adjust_channels(512), adjust_channels(512), stride=1),  # 512×14×14
            DepthwiseSeparableConv(adjust_channels(512), adjust_channels(512), stride=1),  # 512×14×14
            DepthwiseSeparableConv(adjust_channels(512), adjust_channels(512), stride=1),  # 512×14×14
            DepthwiseSeparableConv(adjust_channels(512), adjust_channels(512), stride=1),  # 512×14×14
            DepthwiseSeparableConv(adjust_channels(512), adjust_channels(512), stride=1),  # 512×14×14
            
            # 深度可分离卷积块5
            DepthwiseSeparableConv(adjust_channels(512), adjust_channels(1024), stride=2),  # 1024×7×7
            DepthwiseSeparableConv(adjust_channels(1024), adjust_channels(1024), stride=1),  # 1024×7×7
        )
        
        # 全局平均池化和分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 1024×1×1
            nn.Flatten(),
            nn.Dropout(0.35),  
            nn.Linear(adjust_channels(1024), num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV1().to(device)
    print(summary(model, (3, 224, 224)))