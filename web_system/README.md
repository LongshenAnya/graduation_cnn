# 宠物图像识别展示系统

## 项目简介
这是一个基于Django的猫狗图像识别展示系统，支持用户注册登录、图片上传识别、人工反馈、数据可视化等功能。

## 环境搭建
1. 创建虚拟环境：
   ```bash
   conda create -n pet_web python=3.9 -y
   ```

2. 激活环境：
   ```bash
   conda activate pet_web
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 初始化数据库：
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. 创建管理员：
   ```bash
   python manage.py createsuperuser
   ```

6. 运行项目：
   ```bash
   python manage.py runserver
   ```

## 文件准备
将训练好的模型文件放入 `models/` 目录：
- resnet_catdog.pth
- vgg_catdog.pth
- mobilenet_catdog.pth

**注意：** 模型文件需要您自己准备，本项目不包含训练好的模型文件。

## 录入静态实验数据
1. 访问 http://127.0.0.1:8000/admin
2. 登录管理员账号
3. 在 "Model performances" 表中添加三条记录（ResNet/VGG/MobileNet）：
   - 准确率、推理时间、模型大小等

## 访问地址
- 主页面：http://127.0.0.1:8000
- 图表页：http://127.0.0.1:8000/charts
- 管理后台：http://127.0.0.1:8000/admin

## 功能说明
### 用户认证
- 用户注册和登录
- 未登录用户自动跳转到登录页

### 图片识别
- 支持图片上传和拖拽
- 三种模型选择：ResNet、VGG、MobileNet
- 实时显示识别结果和置信度

### 人工反馈
- 用户可以对识别结果进行纠正
- 反馈数据保存到数据库

### 数据可视化
- 静态数据：实验测试结果（需手动录入）
- 动态数据：实时运行统计（自动更新）

## 注意事项
- 所有模型使用CPU运行
- 首次加载模型可能较慢，之后会缓存
- 上传图片会自动缩放至224x224
- 图表每30秒自动刷新动态数据

## 技术栈
- 后端：Django 4.2
- 前端：Bootstrap 5 + ECharts 5
- 深度学习：PyTorch 2.0 (CPU)
- 数据库：SQLite