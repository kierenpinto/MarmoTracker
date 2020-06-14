#!/usr/bin/python
import cv2 as cv
from position_tools import boxToCentre,findOverPrev
default_location = 'cascade.xml'
class marmoface_detector:
    def __init__(self,location=default_location):
        self.face_cascade = cv.CascadeClassifier(location)

    def runCascade(self,grayscale_image):
        detected_faces = self.face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
        return detected_faces

    def draw_boxes(self,color_image_ptr,detected_faces,color =(255,0,0) ):
        for (x,y,w,h) in detected_faces:
            cv.rectangle(color_image_ptr,(x,y),(x+w,y+h),color,2)
    
    def iterate_faces(self,function,**kwargs):
        ''' pass function in to call on data '''
        output = []
        for (x,y,w,h) in faces:
            output.append(function(*kwargs))
        return output

    def detect(self,grayscale_image):
        fs = self.runCascade(grayscale_image)
        return FaceBoxes(fs)

class FaceBoxes:
    '''Stores Faces For an Image'''
    def __init__(self,boxes_xywh,debug = False):
        ''' Input a gray Image '''
        self._boxes_xywh = boxes_xywh
        self.debug = debug
        # print("faces {} centre {}".format(self._boxes_xywh, face_centres)) if self.debug else None
    
    @property
    def centres(self):
        self._face_centres = boxToCentre(self._boxes_xywh)
        return self._face_centres

    @property
    def boxes(self,mode='xywh'):
        if mode == 'xywh':
            return self._boxes_xywh
        if mode == 'x1y1x2y2':
            pass


class FaceFilter:
    ''' Removes Faces that aren't present for threshold number of frames '''
    cache = []
    def __init__(self,overlap_threshold=0.5,frame_count_threshold=3):
        self.OT = overlap_threshold
        self.FCT = frame_count_threshold
    
    def update(self,boxList):
        ''' Update face filter '''
        self.cache = findOverPrev(self.cache,boxList,self.OT)
        filBoxs = self._filteredBoxes(self.cache,self.FCT)
        # print("filtered > 5 consecutive faces {}".format(filBoxs)) if debug else None
        return filBoxs
    
    @staticmethod
    def _filteredBoxes(boxList,threshold):
        ''' Checks if face is stable in the time domain'''
        func = lambda box: box[4] > threshold
        box_with_count = filter(func, boxList)
        xywh = list(map(lambda box: box[:4],box_with_count))
        return FaceBoxes(xywh)

    def getCache(self,debug=False):
        print("faceCache {}".format(self.cache)) if debug else None
        return self.cache