from django import forms
from .models import Keras_Facenet

class ImageForm(forms.ModelForm):

   class Meta:
      model = Keras_Facenet
      fields = '__all__'