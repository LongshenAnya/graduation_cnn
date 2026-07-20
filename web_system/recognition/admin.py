from django.contrib import admin
from .models import Prediction

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('user', 'image_name', 'model_used', 'prediction', 'confidence', 'inference_time', 'created_at')
    list_filter = ('model_used', 'prediction', 'created_at')
    search_fields = ('user__username', 'image_name')