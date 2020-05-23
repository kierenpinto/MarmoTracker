#!/usr/bin/python

import cv2 as cv
location = 'cascade.xml'
class marmoface_detector:
    def __init__(self):
        self.face_cascade = cv.CascadeClassifier(location)

    def runCascade(self,grayscale_image):
        detected_faces = self.face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
        return detected_faces

    def draw_boxes(self,color_image_ptr,detected_faces):
        for (x,y,w,h) in detected_faces:
            cv.rectangle(color_image_ptr,(x,y),(x+w,y+h),(255,0,0),2)
    
    def iterate_faces(self,function,**kwargs):
        ''' pass function in to call on data '''
        output = []
        for (x,y,w,h) in faces:
            output.append(function(*kwargs))
        return output