import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import time
people_image = cv.imread('messigray.png')
config = rs.config()
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,10)
#enable_stream(self: pyrealsense2.config, stream_type: pyrealsense2.stream, width: int, height: int, format: pyrealsense2.format=format.any, framerate: int=0)
pipe = rs.pipeline()
profile = pipe.start()
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# time.sleep(2)
for i in range(0,100):
	frames = pipe.wait_for_frames()
frames = pipe.wait_for_frames()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())
corrected = color_image[:,:,[2,1,0]]
gray_people = cv.cvtColor(corrected,cv.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(gray_people, 1.3, 5)
print(len(faces))
for (x,y,w,h) in faces:
	print("face found")
	print(len(faces))
	cv.rectangle(corrected,(x,y),(x+w,y+h),(255,0,0),2)
(x,y,w,h) = (303, 28, 174, 174)
cv.rectangle(people_image,(x,y),(x+w,y+h),(255,0,0),2)
cv.namedWindow("corrected", cv.WINDOW_NORMAL)
cv.imshow("corrected",corrected)
cv.imshow('people_image',people_image)
# cv.imwrite('messigray.png',corrected)
cv.waitKey(0)
cv.destroyAllWindows()
pipe.stop()