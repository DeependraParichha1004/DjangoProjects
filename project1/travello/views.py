from django.shortcuts import render,redirect
from .models import Food
from django.contrib.auth.models import auth,User
# Create your views here.

def index(request):

    foods=Food.objects.all()
    return render(request,'index.html',{'foods':foods})
def login(request):
    return render(request,'login.html')

def paneerchilli(request):
    if auth.login():
        return render(request,'chillip.html')
    else:
        return redirect('/')