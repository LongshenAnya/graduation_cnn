from django.db import models
from django.contrib.auth.models import User


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    image_name = models.CharField(max_length=255)
    model_used = models.CharField(max_length=20, choices=[
        ('resnet', 'ResNet'),
        ('vgg', 'VGG'),
        ('mobilenet', 'MobileNet'),
    ])
    task = models.CharField(max_length=10, choices=[
        ('catdog', '猫狗识别'),
        ('breeds', '狗品种识别'),
    ], default='catdog')
    prediction = models.CharField(max_length=50)  # 存储预测结果：'cat'/'dog' 或品种英文名
    display_prediction = models.CharField(max_length=100, default='')  # 用户友好的显示名称
    confidence = models.FloatField()  # 0~1
    inference_time = models.FloatField()  # 秒
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.display_prediction} ({self.confidence:.2f})"