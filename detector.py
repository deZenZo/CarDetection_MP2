import cv2
import sys

cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)

def detectCar(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cars = car_cascade.detectMultiScale(gray,1.1,1)

	if not len(cars):
		return [0,0,0,0]

	for (x,y,w,h) in cars:
		return x,y,w,h

if __name__ == "__main__":
	try:
		image = cv2.imread(sys.argv[1])
		x,y,w,h = detectCar(image)
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.imshow("Detected Car",image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	except :
		print("Error: Invalid Usage")
		print("Usage: detector.py <imagefile>")
		exit()


