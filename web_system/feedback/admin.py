from django.contrib import admin
from .models import Feedback

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('user', 'prediction', 'original_prediction', 'user_correction', 'created_at')
    list_filter = ('original_prediction', 'user_correction', 'created_at')
    search_fields = ('user__username', 'prediction__image_name')