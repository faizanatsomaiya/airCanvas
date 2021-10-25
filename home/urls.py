from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('', views.home, name='homepage'),
    path('function1/', views.function1, name='function1')
]