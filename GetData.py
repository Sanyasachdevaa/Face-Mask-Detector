import cv2
import time


cap = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

while True:
    _ ,frame = cap.read()
    faces = faceDetect.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        count +=1
        name = './images/0/' + str(count) + '.jpg'
        print("creating dataset..."+name)

        cv2.imwrite(name,frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0),2)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)

    if (count == 200):
        break

cap.release()
cv2.destroyAllWindows()