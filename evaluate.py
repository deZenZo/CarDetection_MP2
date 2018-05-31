# -*- coding: utf-8 -*-

import cv2
import sys


cascade_src = 'cars.xml'
img_src = sys.argv[1]

car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    img = cv2.imread(img_src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)    
    
    #x1,y1,x2,y2 = 35,66, 133, 128
    #cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow('car', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()