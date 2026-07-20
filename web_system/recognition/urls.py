from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('api/predict/', views.predict_api, name='predict_api'),
]