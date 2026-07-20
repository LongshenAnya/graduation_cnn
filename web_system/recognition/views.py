from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from .models import Prediction
from .utils.model_loader import load_model
from .utils.preprocess import preprocess_image
from .utils.inference import predict
from .utils.breed_mapper import breed_mapper
from django.contrib import messages


import torch
import os
import sys

print("=" * 50, file=sys.stderr)
print(f"[设备监控] 环境变量 CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}", file=sys.stderr)
print(f"[设备监控] CUDA 是否可用: {torch.cuda.is_available()}", file=sys.stderr)
if torch.cuda.is_available():
    print(f"[设备监控] 检测到 {torch.cuda.device_count()} 块 GPU", file=sys.stderr)
    print(f"[设备监控] 正在使用 GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
else:
    print(f"[设备监控] 未检测到 GPU，将使用 CPU", file=sys.stderr)
print("=" * 50, file=sys.stderr)


@login_required
def home_view(request):
    if request.method == 'POST':
        # 处理图片上传和识别
        image_file = request.FILES.get('image')
        model_name = request.POST.get('model')
        
        if not image_file:
            messages.error(request, '请选择图片文件')
            return redirect('home')
        
        if model_name not in ['resnet', 'vgg', 'mobilenet']:
            messages.error(request, '请选择有效的模型')
            return redirect('home')
        
        try:
            # 加载模型
            model = load_model(model_name)
            
            # 预处理图片
            image_tensor = preprocess_image(image_file)
            
            # 执行推理
            predicted_class, confidence, inference_time = predict(image_tensor, model)
            
            # 保存到数据库
            prediction = Prediction.objects.create(
                user=request.user,
                image=image_file,
                image_name=image_file.name,
                model_used=model_name,
                prediction=predicted_class,
                confidence=confidence,
                inference_time=inference_time,
            )
            
            # 传递结果到模板
            context = {
                'prediction': prediction,
                'confidence_percent': confidence * 100,
                'inference_time_ms': inference_time * 1000,
            }
            return render(request, 'recognition/index.html', context)
            
        except Exception as e:
            messages.error(request, f'识别失败: {str(e)}')
            return redirect('home')
    
    return render(request, 'recognition/index.html')

@csrf_exempt
@require_POST
@login_required
def predict_api(request):
    """
    API接口：图片识别
    """
    try:
        image_file = request.FILES.get('image')
        model_name = request.POST.get('model_name')
        task = request.POST.get('task', 'catdog')  # 默认猫狗识别

        if not image_file or model_name not in ['resnet', 'vgg', 'mobilenet']:
            return JsonResponse({'success': False, 'error': '无效的请求参数'})

        if task not in ['catdog', 'breeds']:
            return JsonResponse({'success': False, 'error': '无效的任务类型'})

        # 根据任务确定类别名称列表
        if task == 'catdog':
            class_names = ['cat', 'dog']
        elif task == 'breeds':
            class_names = breed_mapper.get_breed_list('en')  # 英文品种名称列表

        # 加载模型
        model = load_model(model_name, task)

        # 预处理图片
        image_tensor = preprocess_image(image_file)

        # 执行推理
        predicted_class, confidence, inference_time = predict(image_tensor, model, class_names)

        # 确定显示名称
        if task == 'catdog':
            display_prediction = predicted_class  # 'cat' 或 'dog'
        elif task == 'breeds':
            display_prediction = breed_mapper.get_display_name(predicted_class, 'en')

        # 保存到数据库
        prediction = Prediction.objects.create(
            user=request.user,
            image=image_file,
            image_name=image_file.name,
            model_used=model_name,
            task=task,
            prediction=predicted_class,
            display_prediction=display_prediction,
            confidence=confidence,
            inference_time=inference_time,
        )

        return JsonResponse({
            'success': True,
            'result': predicted_class,
            'display_result': display_prediction,
            'confidence': round(confidence, 4),
            'time': round(inference_time, 4),
            'prediction_id': prediction.id,
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})