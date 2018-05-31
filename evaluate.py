
import cv2
import time
#import sys


labels = open("labels.csv","r")
cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    line = labels.readline()
    if not line:
        exit()

    img_file,x1,y1,x2,y2 = line.replace("\n","").split(",")

    img = cv2.imread("dataset/"+img_file)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
   
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)    
    
    cv2.rectangle(img, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    cv2.imshow('car', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

