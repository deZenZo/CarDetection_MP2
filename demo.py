
import cv2
import time
#import sys


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou





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
        break 
    cv2.rectangle(img, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)


    ground_truth = (int(x1),int(y1),int(x2),int(y2))
    predicted = (x,y,w,h)
    iou = bb_intersection_over_union(ground_truth, predicted)
    cv2.putText(img, "IoU: {:.4f}".format(iou), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    print("{}: {:.4f}".format(img_file, iou))
    cv2.imshow('car', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

