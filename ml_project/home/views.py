from django.shortcuts import render,redirect
from keras_facenet import FaceNet
from .models import Keras_Facenet
import numpy as np
import joblib
from .forms import ImageForm
import psycopg2
from django.http import StreamingHttpResponse
from django.core.files.storage import FileSystemStorage
# Create your views here.
def home(request):
    return render(request,'index.html')

def image(request):
    return render(request,'image.html')

def form(request):
    if request.method=='POST':
        form=ImageForm(request.POST,request.FILES)
        if form.is_valid():
            form.person_name = form.cleaned_data.get('person_name')
            form.image = form.cleaned_data.get('image')
            form.model_name = form.cleaned_data.get('model_name')
            form.save()
            person_name = form.person_name
            # image = form.image
            img = 'data/val/'+str(form.person_name)+'/'+str(form.image)

            model_name = form.model_name
            keras_class = Keras_Facenet()
            facenet = FaceNet()
            face = keras_class.face_extract(img)
            facenew = np.expand_dims(face, axis=0)
            face_emb = facenet.embeddings(list(facenew))
            model_svm = joblib.load('finalized_model.sav')
            encoder = joblib.load('out_encoder.sav')
            yhat_face = model_svm.predict(face_emb)
            yhat_prob = model_svm.predict_proba(face_emb)
            name = encoder.inverse_transform([yhat_face[0]])
            # upload = request.FILES['result']
            # fss = FileSystemStorage()
            # file = fss.save(upload.name, upload)
            # file_url = fss.url(file)

            details = [yhat_face, np.max(yhat_prob) * 100, name, image]
            content=[form.person_name,form.image,form.model_name]
            return render(request,'result.html',{'details':details})
    form = ImageForm()
    return render(request,'form.html',{'form':form})
    # if request.method=='POST':
    #     form = ImageForm(request.POST, request.FILES)
    #     if form.is_valid():
    #         form.save()
    #         return redirect("result")
    #     # person_name = request.POST.get('person_name')
    #     # img = request.POST.get('img')
    #     # model_name = request.POST.get('model_name')
    #     # person_name = request.POST.get('person_name')
    #     # img_data = 'data/val/' + person_name + '/' + request.POST.get('img')
    #     # # model_name = request.POST.get('model_name')
    #     # keras_class = Keras_Facenet()
    #     # facenet = FaceNet()
    #     # face = keras_class.face_extract(img_data)
    #     # facenew = np.expand_dims(face, axis=0)
    #     # face_emb = facenet.embeddings(list(facenew))
    #     # model_svm = joblib.load('finalized_model.sav')
    #     # encoder = joblib.load('out_encoder.sav')
    #     # yhat_face = model_svm.predict(face_emb)
    #     # yhat_prob = model_svm.predict_proba(face_emb)
    #     # name = encoder.inverse_transform([yhat_face[0]])
    #     # upload = request.FILES['form']
    #     # fss = FileSystemStorage()
    #     # file = fss.save(upload.name, upload)
    #     # file_url = fss.url(file)
    #     # details = [yhat_face, np.max(yhat_prob) * 100, img]
    #     # model = Keras_Facenet(person_name =person_name ,image=img,model_name=model_name)
    #     # model.save()
    #     print('model_saved')
    # else:
    #     form=ResumeForm
    # return render(request,'result.html')
def record(request):

    records=Keras_Facenet.objects.values() #all() not working
    return render(request,'record.html',{'records':records})

def result(request):
    person_name=request.POST.get('person_name')
    image = request.POST.get('img')
    img='data/val/'+person_name+'/' + request.POST.get('img')

    model_name=request.POST.get('model_name')
    keras_class = Keras_Facenet()
    facenet = FaceNet()
    face = keras_class.face_extract(img)
    facenew = np.expand_dims(face, axis=0)
    face_emb = facenet.embeddings(list(facenew))
    model_svm=joblib.load('finalized_model.sav')
    encoder=joblib.load('out_encoder.sav')
    yhat_face = model_svm.predict(face_emb)
    yhat_prob = model_svm.predict_proba(face_emb)
    name=encoder.inverse_transform([yhat_face[0]])
    # upload = request.FILES['result']
    # fss = FileSystemStorage()
    # file = fss.save(upload.name, upload)
    # file_url = fss.url(file)

    details=[yhat_face,np.max(yhat_prob)*100,name,img]
    # model1 = Keras_Facenet(person_name=person_name, image='pics/'+image, model_name=model_name)
    # model1.save()
    return render(request,'result.html')

def image(request):
    return render(request,'image.html')
#
# def image(request):
#     import cv2
#     import os
#     cap = cv2.VideoCapture(0)
#     res, image = cap.read()
#     path = 'accounts/'
#     if res:
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         cv2.imwrite('images/image_3.jpg', image)
#         print('image saved successfully')
#         return StreamingHttpResponse(cv2.imshow('image', image))
#
#     else:
#         return render(request,'camera')


from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
def camera(request):
    return render(request,'Camera.html')
# @gzip.gzip_page
# def camera(request):
#     try:
#         cam = VideoCamera(0)
#         print('success')
#         # return HttpResponse("hi")
#         return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
#     except:
#         pass
#     return render(request, 'camera.html')

#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):

        self.video.release()

    def get_image(self):
        return self.frame

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')
