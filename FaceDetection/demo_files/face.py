import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('./cascade.xml')
# people_image = cv.imread('human2.jpg')
# people_image = cv.imread('myface.png')
people_image = cv.imread('face12.jpg')
gray_people = cv.cvtColor(people_image,cv.COLOR_BGR2GRAY)
print(people_image.shape)
print(type(people_image))
faces = face_cascade.detectMultiScale(gray_people, 1.3, 5)
print(len(faces))
for (x,y,w,h) in faces:
	cv.rectangle(people_image,(x,y),(x+w,y+h),(255,0,0),2)
	# roi_gray = gray_people[y:y+h, x:x+w]
	# roi_color = people_image[y:y+h, x:x+w]

cv.imshow('people_image',people_image)
cv.waitKey(0)
cv.destroyAllWindows()