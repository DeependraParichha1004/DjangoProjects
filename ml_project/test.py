from keras_facenet import FaceNet
import cv2
import matplotlib.pyplot as plt
# import joblib
import numpy as np
# from .models import Keras_Facenet
# model=joblib.load('finalized_model.sav')
facenet=FaceNet()
# # facenet('data/train/','data/val/','data/val/Deependra/WhatsApp Image 2022-10-03 at 5.42.53 PM (2).jpeg')
# model_keras=Keras_Facenet()
filename='data/val/Deependra/WhatsApp Image 2022-10-03 at 5.42.53 PM (2).jpeg'
# facenet=FaceNet()
# face = facenet.face_extract(filename)
arr=cv2.imread(filename)
print(cv2.imshow('image',arr+221))
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
# facenew = np.expand_dims(face, axis=0)
# face_emb = facenet.embeddings(list(facenew))