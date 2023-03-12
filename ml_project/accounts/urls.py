
from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('registration', views.registration,name='registration'),
    path('login',views.login,name='login'),
    path('logout',views.logout,name='logout'),
    path('models',views.models,name='models'),
]