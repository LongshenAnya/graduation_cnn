from django.db import models
from django.contrib.auth.models import User
from recognition.models import Prediction


class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE)
    original_prediction = models.CharField(max_length=50)  # 原始预测结果
    display_original = models.CharField(max_length=100, default='')   # 原始预测显示名称
    user_correction = models.CharField(max_length=50)     # 用户纠正结果
    display_correction = models.CharField(max_length=100, default='') # 用户纠正显示名称
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'prediction')  # 一个用户对一个预测只能反馈一次

    def __str__(self):
        return f"{self.user.username} - {self.display_original} -> {self.display_correction}"