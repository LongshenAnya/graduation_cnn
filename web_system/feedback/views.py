from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
import json
from .models import Feedback
from recognition.models import Prediction
from recognition.utils.breed_mapper import breed_mapper

@csrf_exempt
@require_http_methods(["GET", "POST"])
@login_required
def feedback_api(request):
    """
    API接口：提交反馈
    """
    try:
        data = json.loads(request.body)
        prediction_id = data.get('prediction_id')
        correction = data.get('correction')

        if not prediction_id or not correction:
            return JsonResponse({'success': False, 'error': '无效的请求参数'})

        # 获取预测记录
        prediction = Prediction.objects.filter(id=prediction_id, user=request.user).first()
        if not prediction:
            return JsonResponse({'success': False, 'error': '预测记录不存在'})

        # 验证纠正结果
        if prediction.task == 'catdog':
            if correction not in ['cat', 'dog']:
                return JsonResponse({'success': False, 'error': '无效的纠正结果'})
            display_correction = correction  # cat/dog直接使用
        elif prediction.task == 'breeds':
            # 验证品种名称是否存在
            if breed_mapper.get_breed_by_name(correction, 'en') is None:
                return JsonResponse({'success': False, 'error': '无效的品种名称'})
            display_correction = breed_mapper.get_display_name(correction, 'en')
        else:
            return JsonResponse({'success': False, 'error': '未知的任务类型'})

        # 检查是否已反馈
        if Feedback.objects.filter(user=request.user, prediction=prediction).exists():
            return JsonResponse({'success': False, 'error': '已提交过反馈'})

        # 保存反馈
        feedback = Feedback.objects.create(
            user=request.user,
            prediction=prediction,
            original_prediction=prediction.prediction,
            display_original=prediction.display_prediction,
            user_correction=correction,
            display_correction=display_correction,
        )

        return JsonResponse({'success': True})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_http_methods(["GET", "POST"])
@login_required
def stats_dynamic_api(request):
    """
    API接口：获取动态统计数据
    """
    try:
        from django.db.models import Count, Avg
        from django.utils import timezone
        import datetime
        
        # 今日识别次数 (只统计猫狗分类任务)
        today = timezone.now().date()
        today_count = Prediction.objects.filter(created_at__date=today, task='catdog').count()
        
        # 最近10次识别置信度 (只统计猫狗分类任务)
        recent_predictions = Prediction.objects.filter(task='catdog').order_by('-created_at')[:10]
        recent_confidences = [pred.confidence for pred in recent_predictions]
        
        # 三个模型平均推理耗时 (只统计猫狗分类任务)
        model_times = {}
        for model in ['resnet', 'vgg', 'mobilenet']:
            avg_time = Prediction.objects.filter(model_used=model, task='catdog').aggregate(avg_time=Avg('inference_time'))['avg_time']
            model_times[model] = round(avg_time * 1000, 2) if avg_time else 0  # 转换为毫秒
        
        # 猫/狗识别比例 (只统计猫狗分类任务)
        cat_dog_ratio = Prediction.objects.filter(task='catdog').values('prediction').annotate(count=Count('prediction'))
        ratio_dict = {item['prediction']: item['count'] for item in cat_dog_ratio}
        
        return JsonResponse({
            'today_count': today_count,
            'recent_confidences': recent_confidences,
            'model_times': model_times,
            'cat_dog_ratio': ratio_dict,
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_http_methods(["GET", "POST"])
@login_required
def history_api(request):
    """
    API接口：获取历史记录
    """
    try:
        from django.core.paginator import Paginator
        page = int(request.POST.get('page', 1))
        
        predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
        paginator = Paginator(predictions, 10)
        page_obj = paginator.get_page(page)
        
        history_data = []
        for pred in page_obj:
            feedback = Feedback.objects.filter(prediction=pred).first()
            feedback_status = f"已纠正为{feedback.user_correction}" if feedback else "未反馈"
            
            history_data.append({
                'id': pred.id,
                'image_url': pred.image.url,
                'created_at': pred.created_at.strftime('%Y-%m-%d %H:%M'),
                'model_used': pred.model_used,
                'prediction': pred.prediction,
                'confidence': round(pred.confidence * 100, 2),
                'feedback_status': feedback_status,
            })
        
        return JsonResponse({
            'history': history_data,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
            'current_page': page_obj.number,
            'total_pages': paginator.num_pages,
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_http_methods(["GET", "POST"])
@login_required
def user_stats_api(request):
    """
    API接口：用户统计
    """
    try:
        total_count = Prediction.objects.filter(user=request.user).count()
        join_date = request.user.date_joined.strftime('%Y-%m-%d')
        
        return JsonResponse({
            'total_count': total_count,
            'join_date': join_date,
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})