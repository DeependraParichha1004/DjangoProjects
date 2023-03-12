from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('', views.home,name='home'),
    path('form',views.form,name='form'),
    path('result',views.result,name='result'),
    path('record',views.record,name='record'),
    path('camera',views.camera,name='camera'),
    path('image',views.image,name='image'),
    path('video/', views.video_feed, name='video_feed'),
]