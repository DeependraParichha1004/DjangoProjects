import cv2
import os
cap=cv2.VideoCapture(0)
res,image=cap.read()
path='accounts/'
if res:
    cv2.imshow('image',image)

    cv2.imwrite('images/.jpg',image)
    print('image saved successfully')
else:
    print('image is not saved')
cv2.waitKey(0)
cv2.destroyAllWindows()

_,jpeg=cv2.imencode('.jpg',image)
byte=jpeg.tobytes()
# print(byte)
print(image)