from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import ModelPerformance
from recognition.models import Prediction
from feedback.models import Feedback
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import date
from recognition.utils.breed_mapper import breed_mapper

@login_required
def charts_view(request):
    return render(request, 'visualization/charts.html')

@login_required
def performance_api(request):
    """
    API接口：获取模型性能数据，返回结构化对象（用于 /api/performance/）
    参数：task=catdog 或 task=dogbreed
    """
    try:
        task = request.GET.get('task', 'catdog')
        if task not in ['catdog', 'dogbreed', 'breeds']:
            return JsonResponse({'error': 'Invalid task parameter. Use "catdog" or "dogbreed"'}, status=400)

        # 兼容breeds字段存储使用
        if task == 'dogbreed':
            query_task = 'breeds'
        else:
            query_task = task

        performances = ModelPerformance.objects.filter(task=query_task)
        result = {'resnet': {}, 'vgg': {}, 'mobilenet': {}}
        for perf in performances:
            result[perf.model_name] = {
                'accuracy': perf.accuracy,
                'inference_time': perf.inference_time,
                'model_size': perf.model_size,
            }

        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def daily_stats_api(request):
    """
    API接口：获取今日统计数据
    参数：task=catdog 或 task=dogbreed
    """
    try:
        task = request.GET.get('task', 'catdog')
        if task not in ['catdog', 'dogbreed']:
            return JsonResponse({'error': 'Invalid task parameter. Use "catdog" or "dogbreed"'}, status=400)

        today = date.today()
        query_task = 'breeds' if task == 'dogbreed' else task

        # 今日预测数据
        predictions = Prediction.objects.filter(
            task=query_task,
            created_at__date=today
        )

        # 总识别次数
        total_count = predictions.count()

        # 模型使用次数
        model_counts = predictions.values('model_used').annotate(count=Count('model_used')).order_by('model_used')
        model_stats = {item['model_used']: item['count'] for item in model_counts}

        result = {
            'total_count': total_count,
            'resnet_count': model_stats.get('resnet', 0),
            'vgg_count': model_stats.get('vgg', 0),
            'mobilenet_count': model_stats.get('mobilenet', 0),
        }

        if task == 'catdog':
            # 猫狗比例 - 统计所有历史数据
            all_catdog_predictions = Prediction.objects.filter(task='catdog')
            cat_dog_counts = all_catdog_predictions.values('prediction').annotate(count=Count('prediction')).order_by('prediction')
            cat_count = next((item['count'] for item in cat_dog_counts if item['prediction'] == 'cat'), 0)
            dog_count = next((item['count'] for item in cat_dog_counts if item['prediction'] == 'dog'), 0)
            result['cat_count'] = cat_count
            result['dog_count'] = dog_count
        else:
            # 热门品种Top5 - 统计所有历史数据
            all_breed_predictions = Prediction.objects.filter(task='breeds')
            breed_counts = all_breed_predictions.values('prediction').annotate(count=Count('prediction')).order_by('-count')[:5]
            top_breeds = []
            for item in breed_counts:
                breed_name = item['prediction']
                try:
                    # 尝试获取品种的中文名
                    breed_info = breed_mapper.get_breed_by_name(breed_name, 'en')
                    if breed_info:
                        display_name = f"{breed_info['cn']}\n({breed_info['en']})"
                    else:
                        display_name = breed_name
                except:
                    display_name = breed_name
                top_breeds.append({'breed': display_name, 'count': item['count']})
            result['top_breeds'] = top_breeds

        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# 保留旧静态接口兼容（可选）
@login_required
def stats_static_api(request):
    """历史兼容接口"""
    return performance_api(request)

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from recognition.utils.breed_mapper import breed_mapper

@csrf_exempt
@require_http_methods(["GET", "POST"])
@login_required
def stats_dynamic_api_breed(request):
    """
    API接口：获取狗品种识别动态统计数据
    """
    try:
        from django.db.models import Count, Avg
        from django.utils import timezone
        import datetime

        # 今日品种识别次数（task=breeds）
        today = timezone.now().date()
        today_count = Prediction.objects.filter(
            created_at__date=today,
            task='breeds'
        ).count()

        # 最近10次品种识别置信度
        recent_predictions = Prediction.objects.filter(
            task='breeds'
        ).order_by('-created_at')[:10]
        recent_confidences = [pred.confidence for pred in recent_predictions]

        # 三个模型在品种识别上的平均推理耗时
        model_times = {}
        for model in ['resnet', 'vgg', 'mobilenet']:
            avg_time = Prediction.objects.filter(
                model_used=model,
                task='breeds'
            ).aggregate(avg_time=Avg('inference_time'))['avg_time']
            model_times[model] = round(avg_time * 1000, 2) if avg_time else 0  # 转换为毫秒

        # 最近7天识别次数最多的前10个品种
        seven_days_ago = timezone.now() - datetime.timedelta(days=7)
        top_breeds_query = Prediction.objects.filter(
            task='breeds',
            created_at__gte=seven_days_ago
        ).values('prediction').annotate(count=Count('prediction')).order_by('-count')[:10]

        top_breeds = []
        for item in top_breeds_query:
            breed_name = item['prediction']
            breed_info = breed_mapper.get_breed_by_name(breed_name, 'en')
            if breed_info:
                display_name = breed_info['cn']  # 中文名称用于显示
                full_name = f"{breed_info['cn']} ({breed_info['en']})"  # 完整名称用于工具提示
            else:
                display_name = breed_name
                full_name = breed_name
            top_breeds.append({
                'breed': display_name,
                'full_name': full_name,
                'count': item['count']
            })

        return JsonResponse({
            'today_count': today_count,
            'recent_confidences': recent_confidences,
            'model_times': model_times,
            'top_breeds': top_breeds,
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def confidence_overall_api(request):
    """
    API接口：获取所有模型在指定任务上的整体置信度统计
    参数：task=catdog 或 task=dogbreed
    从 ModelConfidenceStats 获取 avg_confidence，从 ModelPerformance 获取 accuracy
    """
    try:
        from .models import ModelConfidenceStats, ModelPerformance
        task = request.GET.get('task', 'catdog')

        # 验证task参数
        if task not in ['catdog', 'dogbreed']:
            return JsonResponse({'error': 'Invalid task parameter. Use "catdog" or "dogbreed"'}, status=400)

        # 从 ModelConfidenceStats 获取置信度数据
        confidence_stats = ModelConfidenceStats.objects.filter(task=task)

        # 从 ModelPerformance 获取准确率数据
        # task 映射：catdog -> catdog, dogbreed -> breeds
        perf_task = 'breeds' if task == 'dogbreed' else task
        performance_stats = ModelPerformance.objects.filter(task=perf_task)

        # 构建性能数据字典
        performance_dict = {perf.model_name: perf.accuracy for perf in performance_stats}

        # 构建结果
        data = {}
        for stat in confidence_stats:
            data[stat.model_name] = {
                'accuracy': performance_dict.get(stat.model_name),  # 从 ModelPerformance 获取
                'avg_confidence': stat.avg_confidence
            }

        return JsonResponse(data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def confidence_byclass_api(request):
    """
    API接口：获取所有模型在指定任务上的各类别置信度统计
    参数：task=catdog 或 task=dogbreed
    """
    try:
        from .models import ClassConfidenceStats
        task = request.GET.get('task', 'catdog')

        # 验证task参数
        if task not in ['catdog', 'dogbreed']:
            return JsonResponse({'error': 'Invalid task parameter. Use "catdog" or "dogbreed"'}, status=400)

        stats = ClassConfidenceStats.objects.filter(task=task).order_by('class_name')
        data = {'classes': [], 'resnet': {}, 'vgg': {}, 'mobilenet': {}}

        # 收集所有类别
        class_set = set()
        for stat in stats:
            class_set.add(stat.class_name)

        # 对类别进行排序
        if task == 'catdog':
            # 猫狗分类固定顺序
            data['classes'] = ['猫', '狗']
        else:
            # 狗品种识别按名称排序，并转换为中英文格式
            sorted_classes = sorted(class_set)
            data['classes'] = []
            for class_name in sorted_classes:
                # 尝试从breed_mapper获取中文名称
                breed_info = breed_mapper.get_breed_by_name(class_name, 'en')
                if breed_info:
                    display_name = f"{breed_info['cn']}/{class_name}"
                else:
                    display_name = class_name
                data['classes'].append(display_name)

        # 填充数据
        for stat in stats:
            model_name = stat.model_name
            class_name = stat.class_name

            # 对于品种识别，使用中英文格式作为key
            if task == 'dogbreed':
                breed_info = breed_mapper.get_breed_by_name(class_name, 'en')
                if breed_info:
                    key = f"{breed_info['cn']}/{class_name}"
                else:
                    key = class_name
            else:
                key = class_name

            data[model_name][key] = stat.accuracy

        return JsonResponse(data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)