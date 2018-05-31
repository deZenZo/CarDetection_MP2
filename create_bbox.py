import cv2
import sys

img = cv2.imread(sys.argv[1])

x1,y1,x2,y2 = 225,130,390,260
cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("annotate",img)
cv2.waitKey(0)
cv2.removeAllWindows()