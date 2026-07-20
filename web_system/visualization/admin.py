from django.contrib import admin
from django import forms
from import_export import resources
from import_export.admin import ImportExportModelAdmin
from .models import ModelPerformance, ModelConfidenceStats, ClassConfidenceStats
from recognition.utils.breed_mapper import BreedMapper

# 获取品种映射器实例
breed_mapper = BreedMapper()

# 猫狗分类选项
CAT_DOG_CHOICES = [
    ('猫', '猫'),
    ('狗', '狗'),
]

# 狗品种选项（动态生成）
def get_breed_choices():
    breeds = breed_mapper.get_breed_list('en')
    return [(breed, breed) for breed in breeds]

# 创建资源类
class ModelPerformanceResource(resources.ModelResource):
    class Meta:
        model = ModelPerformance
        import_id_fields = ['model_name', 'task']
        fields = ('model_name', 'task', 'accuracy', 'inference_time', 'model_size', 'dataset', 'notes')
        skip_unchanged = True

class ModelConfidenceStatsResource(resources.ModelResource):
    class Meta:
        model = ModelConfidenceStats
        import_id_fields = ['model_name', 'task']
        fields = ('model_name', 'task', 'total_samples', 'avg_confidence')
        skip_unchanged = True

# 自定义表单
class ClassConfidenceStatsForm(forms.ModelForm):
    class Meta:
        model = ClassConfidenceStats
        fields = '__all__'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 根据task动态设置class_name的choices
        if 'task' in self.data:
            task = self.data.get('task')
        elif self.instance and self.instance.pk:
            task = self.instance.task
        else:
            task = None
        
        if task == 'catdog':
            self.fields['class_name'] = forms.ChoiceField(
                choices=CAT_DOG_CHOICES,
                label="类别名称"
            )
        elif task == 'dogbreed':
            self.fields['class_name'] = forms.ChoiceField(
                choices=get_breed_choices(),
                label="类别名称"
            )

# 自定义资源类，支持导入时的类别验证
class ClassConfidenceStatsResource(resources.ModelResource):
    class Meta:
        model = ClassConfidenceStats
        import_id_fields = ['model_name', 'task', 'class_name']
        fields = ('model_name', 'task', 'class_name', 'sample_count', 'accuracy', 'avg_confidence')
        skip_unchanged = True
        report_skipped = False
    
    def before_import_row(self, row, **kwargs):
        """导入前验证类别名称"""
        task = row.get('task')
        class_name = row.get('class_name')
        
        if task == 'catdog':
            # 转换英文到中文
            if class_name == 'cat':
                row['class_name'] = '猫'
            elif class_name == 'dog':
                row['class_name'] = '狗'
            else:
                raise ValueError(f"猫狗分类任务的类别名称必须是'cat'或'dog'，当前值: {class_name}")
        elif task == 'dogbreed':
            valid_breeds = breed_mapper.get_breed_list('en')
            if class_name not in valid_breeds:
                raise ValueError(f"狗品种识别任务的类别名称无效: {class_name}")
        
        return super().before_import_row(row, **kwargs)

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(ImportExportModelAdmin):
    resource_class = ModelPerformanceResource
    list_display = ('model_name', 'task', 'accuracy', 'inference_time', 'model_size', 'dataset', 'created_at')
    list_filter = ('model_name', 'task', 'dataset', 'created_at')
    search_fields = ('model_name', 'task', 'dataset', 'notes')

@admin.register(ModelConfidenceStats)
class ModelConfidenceStatsAdmin(ImportExportModelAdmin):
    resource_class = ModelConfidenceStatsResource
    list_display = ('model_name', 'task', 'total_samples', 'avg_confidence', 'created_at')
    list_filter = ('model_name', 'task', 'created_at')
    search_fields = ('model_name', 'task')

@admin.register(ClassConfidenceStats)
class ClassConfidenceStatsAdmin(ImportExportModelAdmin):
    resource_class = ClassConfidenceStatsResource
    form = ClassConfidenceStatsForm
    list_display = ('model_name', 'task', 'class_name', 'sample_count', 'accuracy', 'avg_confidence', 'created_at')
    list_filter = ('model_name', 'task', 'created_at')
    search_fields = ('model_name', 'task', 'class_name')