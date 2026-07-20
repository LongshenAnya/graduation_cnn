from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from recognition.models import Prediction
from feedback.models import Feedback
from django.db.models import Q


def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, '注册成功！')
            return redirect('home')
        else:
            messages.error(request, '注册失败，请检查输入。')
    else:
        form = UserCreationForm()
    return render(request, 'users/register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, '用户名或密码错误。')
    return render(request, 'users/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def profile_view(request):
    user = request.user
    # 获取用户统计
    total_count = Prediction.objects.filter(user=user).count()
    join_date = user.date_joined.strftime('%Y-%m-%d')
    
    # 获取历史记录
    predictions = Prediction.objects.filter(user=user).order_by('-created_at')
    paginator = Paginator(predictions, 10)  # 每页10条
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # 为每条记录添加反馈状态
    for pred in page_obj:
        feedback = Feedback.objects.filter(prediction=pred).first()
        if feedback:
            pred.feedback_status = f"已纠正为{feedback.user_correction}"
        else:
            pred.feedback_status = "未反馈"
    
    context = {
        'user': user,
        'total_count': total_count,
        'join_date': join_date,
        'page_obj': page_obj,
    }
    return render(request, 'users/profile.html', context)