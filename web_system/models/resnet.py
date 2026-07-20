import torch
from torch import nn
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLu(self.bn1(self.conv1(x)))
        y = self.ReLu(self.bn2(self.conv2(y)))

        if self.conv3:
            x = self.conv3(x)
        y = self.ReLu(y + x)
        return y
    
class ResNet18(nn.Module):
    def __init__(self, Residual, num_classes=10):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            Residual(64, 64, use_1conv=False, strides=1),
            Residual(64, 64, use_1conv=False, strides=1)
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, strides=2),
            Residual(128, 128, use_1conv=False, strides=1)
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, strides=2),
            Residual(256, 256, use_1conv=False, strides=1)
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, strides=2),
            Residual(512, 512, use_1conv=False, strides=1)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
        #模型初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, Residual, num_classes=10):
        super(ResNet34, self).__init__()
        
        # 第一层保持不变
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # b2: 3个残差块（ResNet18是2个）
        self.b2 = nn.Sequential(
            Residual(64, 64, use_1conv=False, strides=1),
            Residual(64, 64, use_1conv=False, strides=1),
            Residual(64, 64, use_1conv=False, strides=1)  # 新增的第3个块
        )
        
        # b3: 4个残差块（ResNet18是2个）
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, strides=2),   # 第一个块下采样
            Residual(128, 128, use_1conv=False, strides=1),
            Residual(128, 128, use_1conv=False, strides=1),
            Residual(128, 128, use_1conv=False, strides=1)   # 新增的第4个块
        )
        
        # b4: 6个残差块（ResNet18是2个）
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, strides=2),   # 第一个块下采样
            Residual(256, 256, use_1conv=False, strides=1),
            Residual(256, 256, use_1conv=False, strides=1),
            Residual(256, 256, use_1conv=False, strides=1),
            Residual(256, 256, use_1conv=False, strides=1),
            Residual(256, 256, use_1conv=False, strides=1)   # 新增的第6个块
        )
        
        # b5: 3个残差块（ResNet18是2个）
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, strides=2),   # 第一个块下采样
            Residual(512, 512, use_1conv=False, strides=1),
            Residual(512, 512, use_1conv=False, strides=1)   # 新增的第3个块
        )
        
        # 分类器保持不变
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 模型初始化（与ResNet18相同）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34(Residual).to(device)
    print(summary(model, (3, 224, 224)))