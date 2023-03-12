from django.db import models

# Create your models here.
class Food(models.Model):
    name = models.CharField(max_length=100)
    img=models.ImageField(upload_to='pics')#it is going to be store in the folder
    price = models.IntegerField()
    offer = models.BooleanField(default=False)