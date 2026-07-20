from django.urls import path
from . import views

urlpatterns = [
    path('feedback/', views.feedback_api, name='feedback_api'),
    path('stats/dynamic/', views.stats_dynamic_api, name='stats_dynamic_api'),
    path('history/', views.history_api, name='history_api'),
    path('user/stats/', views.user_stats_api, name='user_stats_api'),
]