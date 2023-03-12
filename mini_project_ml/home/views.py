from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def home(request):
    img='breakfast/brett-jordan-8xt8-HIFqc8-unsplash.jpg'
    return render(request,'index.html')
