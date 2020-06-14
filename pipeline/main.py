#!/usr/bin/python
from camera import depth_camera
from face_detection import marmoface_detector,FaceFilter
from marmoset_detector import YOLODetector
import cv2 as cv
from position_tools import mapMarmosetToFace
from face_recognition import FaceRecognition
from tracking import Tracker
from MarmosetMovementPlot import PlotSequence, AnimationPlot

debug = False
# Initialise Camera
cam = depth_camera()
cam.start()

# Initialise face detector
face_detector = marmoface_detector()

# Initialise Marmoset Detector
marmoset_detector = YOLODetector()

# Initialise Face Recognition
faceRec = FaceRecognition()

# Initialise Marmoset Movement Plot
# pltSeq3d = PlotSequence()
pltSeq = AnimationPlot()
pltSeq.start()


faceFilter = FaceFilter()
tracker = Tracker(cam)

while True:
    # Get frame
    cam.update_frame()
    color_image = cam.get_color()
    gray_image = cam.to_grey(color_image)
    depth_frame = cam.get_depth()


    # Run face detector
    faces = face_detector.detect(gray_image)
    filBoxs = faceFilter.update(faces.boxes) #filtered in time domain
    print("filtered > 5 consecutive faces {}".format(filBoxs)) if debug else None


    # Run marmoset detector    
    marms = marmoset_detector.detect(color_image)
    print("marmosets {} centre {}".format(marms.boxes,marms.centres)) if debug else None

    # Run face recognition
    faceIds = faceRec.autoCropRec(gray_image,filBoxs.boxes)

    print(faceIds) if len(faceIds)>0 and debug else None #
    # For each marmoset find the closest face.
    # marm_to_face = mapMarmosetToFace(faces.centres,marms.centres)

    # Get 3D Coordinates

    # Tracker:
    marmoset_Labels = tracker.update(filBoxs,faceIds,marms)
    # Draw live feed
    face_detector.draw_boxes(color_image,filBoxs.boxes,color=(0,0,255))
    marmoset_detector.draw_boxes(color_image, marms.boxes,marmoset_Labels)
    marmoset_detector.draw_centre(color_image, marms.centres, cam.deproject(marms.centres))
    cv.imshow('live feed',color_image)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        #print some results
        pass


    # Feed through Kalman Filter


    # Graph
    plotPoints = tracker.getPlotPoints()
    pltSeq.update(plotPoints)
    # pltSeq3d.update(plotPoints)

pltSeq.exit()