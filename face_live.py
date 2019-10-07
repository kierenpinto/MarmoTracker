import pyrealsense2 as rs
import cv2 as cv
import numpy as np
# people_image = cv.imread('messigray.png')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

config = rs.config()
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,15)
config.enable_stream(rs.stream.depth,640,480,rs.format.y16,15)
pipe = rs.pipeline()
pipe.start()

while True:
	# capture frame
	frames = pipe.wait_for_frames()
	color_frame = frames.get_color_frame()
	depth_frame = frames.get_depth_frame()
	depth_image = np.asanyarray(depth_frame.get_data())
	# color_image = np.asanyarray(color_frame.get_data())[:,:,[2,1,0]]
	color_image = np.asanyarray(color_frame.get_data())
	grayscale_image = cv.cvtColor(color_image,cv.COLOR_RGB2GRAY)
	img = cv.cvtColor(color_image,cv.COLOR_RGB2BGR)
	# img = np.copy(color_image)
	faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
	for (x,y,w,h) in faces:
		cv.rectangle(depth_image,(x,y),(x+w,y+h),(255,0,0),2)
		cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		# roi_gray = grayscale_image[y:y+h, x:x+w]
		# roi_color = color_image[y:y+h, x:x+w]
	cv.imshow('people_image',img)
	# cv.imshow('people_image',people_image)
	if cv.waitKey(1) == ord('q'):
		break

cv.destroyAllWindows()