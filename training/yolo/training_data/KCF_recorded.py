#!/usr/bin/env python

'''

Written by Kieren Pinto 5/04/2020

This script takes an input video file and runs it through a KCF filter. 
This KCF filter allows for labelling of the video to train the DarkNet Yolo network(s).
Change the save_directory and video file location when using this file.
Usage:
	q to quit
	s to select a new kcf region
	s then c to cancel that region
	f to fast forward by a second
	b to rewind by a second

'''
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
from KCF_lib import KCF_Tracker

config = rs.config()
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,15)
config.enable_stream(rs.stream.depth,640,480,rs.format.y16,15)
pipe = rs.pipeline()
pipe.start()

#Change save directory here
save_directory = './group/marmosets/'

kcf = KCF_Tracker()

#Change video file here:
vs = cv.VideoCapture('./group/marmosets.m4v')
window_name = 'people_image'
img_no = 0
while vs.isOpened():
	# capture frame
	frame = vs.read()
	img = frame[1]
	original = np.copy(img)
	coords = kcf.update(img)
	
	if coords:
		img_no = img_no + 1
		(im_H, im_W) = img.shape[:2]
		save_path = save_directory + str(img_no)
		cv.imwrite(save_path +'.jpg',original)
		height = coords['h']/im_H
		width = coords['w']/im_W
		centreX = (coords['x']+coords['w']/2)/im_W
		centreY = (coords['y']+coords['h']/2)/im_H
		obj_class = 0
		label = "{} {} {} {} {}".format(obj_class,centreX,centreY,width,height)
		labelfile = open(save_path +'.txt','w')
		labelfile.write(label)
		labelfile.close()

	cv.imshow(window_name,img)
	key = cv.waitKey(33)
	if key == ord('q'):
		break
	elif key == ord('s'):
		kcf.choose_roi(img,window_name)
	elif key == ord('b'):
		frameid = vs.get(cv.CAP_PROP_POS_FRAMES)
		new_frame = frameid - 30
		vs.set(cv.CAP_PROP_POS_FRAMES,new_frame)
		print("moved from {} to {}".format(frameid,new_frame))
	elif key == ord('f'):
		frameid = vs.get(cv.CAP_PROP_POS_FRAMES)
		new_frame = frameid + 30
		vs.set(cv.CAP_PROP_POS_FRAMES,new_frame)
		print("moved from {} to {}".format(frameid,new_frame))

cv.destroyAllWindows()
