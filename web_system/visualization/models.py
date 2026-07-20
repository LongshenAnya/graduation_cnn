from django.db import models


class ModelPerformance(models.Model):
    model_name = models.CharField(max_length=20, choices=[
        ('resnet', 'ResNet'),
        ('vgg', 'VGG'),
        ('mobilenet', 'MobileNet'),
    ])
    task = models.CharField(max_length=20, choices=[
        ('catdog', '猫狗分类'),
        ('breeds', '狗品种识别'),
    ], default='catdog', help_text="任务类型")
    accuracy = models.FloatField(help_text="测试准确率，如95.5表示95.5%")
    inference_time = models.FloatField(help_text="平均推理时间（毫秒）")
    model_size = models.FloatField(help_text="模型大小（MB）")
    dataset = models.CharField(max_length=100, help_text="测试数据集名称")
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, help_text="备注信息")

    class Meta:
        ordering = ['model_name']

    def __str__(self):
        return f"{self.model_name} - {self.accuracy}% - {self.dataset}"


class ModelConfidenceStats(models.Model):
    """模型置信度统计表"""
    model_name = models.CharField(max_length=20, choices=[
        ('resnet', 'ResNet'),
        ('vgg', 'VGG'),
        ('mobilenet', 'MobileNet'),
    ])
    task = models.CharField(max_length=20, choices=[
        ('catdog', '猫狗分类'),
        ('dogbreed', '狗品种识别'),
    ], help_text="任务类型")
    total_samples = models.IntegerField(help_text="总样本数")
    # 删除 accuracy 字段，从 ModelPerformance 表关联获取
    avg_confidence = models.FloatField(help_text="整体平均置信度，如95.5表示95.5%")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['model_name', 'task']
        unique_together = ['model_name', 'task']

    def __str__(self):
        return f"{self.model_name} - {self.task} - 置信度:{self.avg_confidence}%"


class ClassConfidenceStats(models.Model):
    """各类别置信度统计表"""
    model_name = models.CharField(max_length=20, choices=[
        ('resnet', 'ResNet'),
        ('vgg', 'VGG'),
        ('mobilenet', 'MobileNet'),
    ])
    task = models.CharField(max_length=20, choices=[
        ('catdog', '猫狗分类'),
        ('dogbreed', '狗品种识别'),
    ], help_text="任务类型")
    class_name = models.CharField(max_length=100, help_text="类别名称")
    sample_count = models.IntegerField(help_text="该类别样本数")
    accuracy = models.FloatField(help_text="该类别准确率，如95.5表示95.5%")
    avg_confidence = models.FloatField(help_text="该类别平均置信度，如95.5表示95.5%")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['model_name', 'task', 'class_name']
        unique_together = ['model_name', 'task', 'class_name']

    def __str__(self):
        return f"{self.model_name} - {self.task} - {self.class_name} - 置信度:{self.avg_confidence}%"