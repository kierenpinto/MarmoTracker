#! /usr/bin/env python
import os
from fisherfaces import FisherFaces
import numpy as np

class_dir_list = ['./test_faces/1/','./test_faces/2/']
class_img_list = ['./test_faces/1/551.jpg-1','./test_faces/2/29.jpg-1']
class FaceRecognition:
    def __init__(self, class_dir_list=class_dir_list,class_img_list=class_img_list):
        instance = FisherFaces()
        for face_class in class_dir_list:
            instance.add_class(face_class)
        instance.load()
        instance.train()
        weights = []
        for face_class in class_img_list:
            weights.append(instance.read_project(face_class))
        self.weights = weights
        self.instance = instance
    
    def recognize(self,face_image):
        
        current_weights = self.instance.project(face_image)
        print(face_image.shape, current_weights.shape, self.weights[0].shape)
        distances = []
        for weight in self.weights:
            distances.append(self.instance.euclidean_distance(current_weights,weight))
        faceId = np.argmin(np.asarray(distances),axis=0)
        return faceId

    def multiRecognise(self,face_images):
        return list(map(self.recognize,face_images))
            
    def autoCropRec(self, gray_image, face_rectangles):
        def recToImg(rec):
            x,y,w,h = rec
            return self.recognize(gray_image[y:y+h,x:x+w])
        return list(map(recToImg,face_rectangles))