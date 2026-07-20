from django.urls import path
from . import views

urlpatterns = [
    path('', views.charts_view, name='charts'),
    path('performance/', views.performance_api, name='performance_api'),
    path('stats/static/', views.stats_static_api, name='stats_static_api'),
    path('stats/dynamic/breed/', views.stats_dynamic_api_breed, name='stats_dynamic_api_breed'),
    path('confidence/overall/', views.confidence_overall_api, name='confidence_overall_api'),
    path('confidence/byclass/', views.confidence_byclass_api, name='confidence_byclass_api'),
    path('daily/stats/', views.daily_stats_api, name='daily_stats_api'),
]