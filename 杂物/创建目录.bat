@echo off
chcp 65001 >nul
echo 正在创建CNN宠物图像识别系统项目结构...
echo.

REM 设置项目根目录
set PROJECT_ROOT=D:\study\graduation_cnn
cd /d %PROJECT_ROOT%

echo 项目根目录: %PROJECT_ROOT%
echo.

REM 1. 创建基础目录
echo 1. 创建基础目录...
mkdir data data\raw data\processed
mkdir models experiments config scripts web_system notebooks

REM 2. 创建raw数据集子目录
echo 2. 创建原始数据集目录...
mkdir data\raw\oxford_pets
mkdir data\raw\stanford_dogs
mkdir data\raw\cats_vs_dogs

REM 3. 创建processed数据集子目录
echo 3. 创建处理后数据集目录...
for %%d in (oxford_pets stanford_dogs cats_vs_dogs) do (
    mkdir data\processed\%%d
    mkdir data\processed\%%d\images
    mkdir data\processed\%%d\train
    mkdir data\processed\%%d\val
    mkdir data\processed\%%d\test
)

REM 4. 创建模型目录结构
echo 4. 创建模型目录...
for %%m in (resnet vgg mobilenet) do (
    mkdir models\%%m
    mkdir models\%%m\weights
)

REM 5. 创建实验日志目录
echo 5. 创建实验记录目录...
for %%m in (resnet vgg mobilenet) do (
    mkdir experiments\logs\%%m
    for %%d in (oxford_pets stanford_dogs cats_vs_dogs) do (
        mkdir experiments\logs\%%m\%%d
    )
)
mkdir experiments\results
mkdir experiments\plots

REM 6. 创建配置文件目录
echo 6. 创建配置目录...
mkdir config\datasets
mkdir config\models

REM 7. 创建Web系统目录
echo 7. 创建Web系统目录...
mkdir web_system\backend
mkdir web_system\frontend
mkdir web_system\frontend\static
mkdir web_system\frontend\templates
mkdir web_system\model_serving

REM 8. 创建Python脚本文件
echo 8. 创建Python脚本文件...

REM 数据相关
echo. > data\dataset_loader.py
echo. > data\__init__.py

REM 模型相关
echo. > models\__init__.py
echo. > models\model_factory.py
echo. > models\train.py
echo. > models\evaluate.py
echo. > models\utils.py

for %%m in (resnet vgg mobilenet) do (
    echo. > models\%%m\__init__.py
    echo. > models\%%m\model.py
)

REM 配置文件
echo. > config\config.yaml
for %%d in (oxford_pets stanford_dogs cats_vs_dogs) do (
    echo. > config\datasets\%%d.yaml
)
for %%m in (resnet vgg mobilenet) do (
    echo. > config\models\%%m.yaml
)

REM 脚本文件
echo. > scripts\preprocess_all_datasets.py
echo. > scripts\train_all_models.py
echo. > scripts\evaluate_all_models.py

REM 9. 创建其他文件
echo 9. 创建其他文件...
echo # 宠物图像识别系统 > README.md
echo. > requirements.txt
echo. > .gitignore

REM 10. 创建笔记本文件
echo 10. 创建笔记本文件...
for %%n in (data_exploration model_training results_analysis) do (
    echo. > notebooks\%%n.ipynb
)

echo.
echo ============================================
echo 项目结构创建完成！
echo 总目录数: 33
echo 总文件数: 24
echo ============================================
echo.
echo 请在以下位置放置你的数据集:
echo 1. D:\study\graduation_cnn\data\raw\oxford_pets\
echo 2. D:\study\graduation_cnn\data\raw\stanford_dogs\
echo 3. D:\study\graduation_cnn\data\raw\cats_vs_dogs\
echo.
echo 按任意键退出...
pause >nul