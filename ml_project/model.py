# # packages
# import os
# import joblib
# import keras_facenet
# import cv2
# import mtcnn
# import pandas as pd
# from keras_facenet import FaceNet
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# from mtcnn.mtcnn import MTCNN
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Normalizer
# from sklearn.svm import SVC
# import django
# from django.db import models
# print('hi')
# img = cv2.imread('lataji_1.jpg')
# # plt.imshow(img,cmap='gray',interpolation='bicubic')
# # plt.show()
#
# model_svc = SVC(kernel='linear', probability=True)
#
#
# class Keras_Facenet(models.Model):
#     def face_extract(self, filename, required_size=(160, 160)):
#         image = Image.open(filename)
#         image = image.convert('RGB')
#         pixels = np.asarray(image)
#         detector = MTCNN()
#         faces = detector.detect_faces(pixels)
#         x, y, width, height = faces[0]['box']
#         x1, y1 = abs(x) + width, abs(y) + height
#         face = pixels[y:y1, x:x1]
#
#         image = Image.fromarray(face)  # use new variable(i.e image/or any other)
#         image = image.resize(required_size)
#         face_array = np.asarray(image)
#         return face_array
#
#     def load_faces(self, dir):
#         faces = list()
#         for subdir in os.listdir(dir):
#             path = dir + subdir
#             face = self.face_extract(path)
#             faces.append(face)
#         return faces
#
#     def create_dataset(self, dir):  # images
#         X = list()
#         Y = list()
#         for subdir in os.listdir(dir):  # subdir==harsh
#             path = dir + subdir + '/'
#             faces = self.load_faces(path)
#             labels = [subdir for i in range(len(faces))]
#             # print("loaded %d sample for class: %s" % (len(faces), subdir))  # print progress
#             X.extend(faces)
#             Y.extend(labels)
#         # return np.asarray(X),np.asarray(Y)
#         return X, Y
#
#     def __call__(self, train_path, test_path, filename):
#
#         facenet = Keras_Facenet()
#         trainx, trainy = facenet.create_dataset(train_path)  # list
#         testx, testy = facenet.create_dataset(test_path)
#         facenet_model = FaceNet()
#         train_emb = facenet_model.embeddings(trainx)
#         print(train_emb)
#         print(train_emb.shape)
#
#         test_emb = facenet_model.embeddings(testx)
#         x_train, y_train = np.asarray(train_emb), np.asarray(trainy)
#         x_test, y_test = np.array(test_emb), np.asarray(testy)
#
#         print("Dataset: train=%d, test=%d" % (train_emb.shape[0], test_emb.shape[0]))
#         # normalize input vectors
#         in_encoder = Normalizer()
#         emdTrainX_norm = in_encoder.transform(train_emb)
#         emdTestX_norm = in_encoder.transform(test_emb)
#         # label encode targets
#         out_encoder = LabelEncoder()
#         out_encoder.fit(y_train)
#         trainy_enc = out_encoder.transform(y_train)
#         testy_enc = out_encoder.transform(y_test)
#         # fit model
#
#         model_svc.fit(emdTrainX_norm, trainy_enc)
#         # predict
#         yhat_train = model_svc.predict(emdTrainX_norm)
#         yhat_test = model_svc.predict(emdTestX_norm)
#         # score
#         score_train = accuracy_score(trainy_enc, yhat_train)
#         score_test = accuracy_score(testy_enc, yhat_test)
#         # summarize
#         print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
#         filename_path = 'finalized_model.sav'
#         joblib.dump(model_svc, filename_path)
#         print('model ran successfully')
#
#         #         def show_image(self,facenet_model,filename):#filename='data/val/Deependra/WhatsApp Image 2022-10-03 at 5.42.53 PM (2).jpeg'
#         face = self.face_extract(filename)
#         facenew = np.expand_dims(face, axis=0)
#         face_emb = facenet_model.embeddings(list(facenew))
#         print('emb_shape', face_emb.shape)
#         # face_emb=np.expand_dims(face_emb,axis=0)
#         yhat_face = model_svc.predict(face_emb)
#         yhat_prob = model_svc.predict_proba(face_emb)
#         name=out_encoder.inverse_transform(testy_enc[yhat_face[0]])
#         print('class is:',name )
#         plt.imshow(face)
#         plt.show()
#
#         person_name=models.CharField(max_length=50)
#         img=models.ImageField()
#         probability=models.FloatField()
#         model_name=models.CharField(max_length=40)
#
#
#
#
# #
# # # inference of a random image
# # from random import choice
# # # select a random face from test set
# # selection = choice([i for i in range(np.asarray(testx).shape[0])])
# # random_face = np.asarray(testx)[selection]
# # random_face_emd = emdTestX_norm[selection]
# # random_face_class = testy_enc[selection]
# # random_face_name = out_encoder.inverse_transform([random_face_class])
# #
# # # prediction for the face
# # samples = np.expand_dims(random_face_emd, axis=0)
# # yhat_class = model_svc.predict(samples)
# # yhat_prob = model_svc.predict_proba(samples)
# # # get name
# # class_index = yhat_class[0]
# # class_probability = yhat_prob[0,class_index] * 100# shape - 2D
# # predict_names = out_encoder.inverse_transform(yhat_class)
# # all_names = out_encoder.inverse_transform([0,1,2,3,4,5])
# # #print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
# # print('Predicted: \n%s \n%s' % (all_names, yhat_prob[0]*100))
# # print('Expected: %s' % random_face_name[0])
# # # plot face
# # plt.imshow(random_face)
# # title = '%s (%.3f)' % (predict_names[0], class_probability)
# # plt.title(title)
# # plt.show()
#
# # facenet=Keras_Facenet()
# # facenet('data/train/','data/val/','data/val/Deependra/WhatsApp Image 2022-10-03 at 5.42.53 PM (1).jpeg')