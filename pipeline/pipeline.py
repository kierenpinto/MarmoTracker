#!/usr/bin/python
from camera import depth_camera
from face_detection import marmoface_detector
from marmoset_detector import YOLODetector
import cv2 as cv
from position_tools import mapMarmosetToFace, boxToCentre
# Initialise Camera
cam = depth_camera()
cam.start()

# Initialise face detector
face_detector = marmoface_detector()

# Initialise Marmoset Detector
marmoset_detector = YOLODetector()
while True:
    # Get frame
    cam.update_frame()
    color_image = cam.get_color()
    gray_image = cam.to_grey(color_image)
    # Run face detector
    detected_faces = face_detector.runCascade(gray_image)
    face_centres = list(map(boxToCentre,detected_faces)) if len(detected_faces)>0 else []
    print("faces {} centre {}".format(detected_faces, face_centres))
    # Run marmoset detector
    marm_box, marm_confid, marm_centre = \
        marmoset_detector.predict_nms(color_image)
    print("marmosets {} centre {}".format(marm_box,marm_centre))
    # For each marmoset find the closest face.
    marm_to_face = mapMarmosetToFace(face_centres,marm_centre)
    # Get 3D Coordinates
    

    # Map marmoset to previous frame
    face_detector.draw_boxes(color_image, detected_faces)
    marmoset_detector.draw_boxes(color_image, marm_box)
    cv.imshow('live feed',color_image)
    if cv.waitKey(1) == ord('q'):
        break
    # Feed through Kalman Filter


    # Graph