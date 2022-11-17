from pyexpat import model
import cv2
from cv2 import threshold
import numpy as np
import pickle
from keras.models import load_model

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold = 0.50
cap = cv2.VideoCapture(0)
cap.set(3,680)
cap.set(4,480)
font = cv2.FONT_HERSHEY_SIMPLEX

model = load_model('TrainingModel.h5')

def processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def get_classname(classNo):
    if classNo == 0:
        return "Mask"

    elif classNo == 1:
        return "Without Mask"

while True:
    suc, Frame = cap.read()
    faces = faceDetect.detectMultiScale(Frame, 1.3,5)
    
    for x , y , w, h in faces:
        crop_img = Frame[y:y+h,x:x+w]
        img = cv2.resize(crop_img, (32,32))
        img = processing(img)
        img = img.reshape(1,32,32,1)

        prediction = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(prediction)

    

        if probabilityValue > threshold:

            if classIndex == 1:
                cv2.rectangle(Frame, (x,y), (x+w, y+h), (0,0,255), 2)
                cv2.rectangle(Frame, (x,y-40), (x+w, y), (0,0,255),-2)
                cv2.putText(Frame, str(get_classname(classIndex)), (x,y-10), font, 0.75, (255,255,255), 1, cv2.LINE_AA)

            print(f"Probability : {str(round(probabilityValue * 100, 2))} %")

    


    cv2.imshow("Results", Frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cap.destroyAllWindows()