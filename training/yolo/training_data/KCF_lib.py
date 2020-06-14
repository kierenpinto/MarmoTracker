# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

class KCF_Tracker:
    # initialize the bounding box coordinates of the object we are going to track
    initBB = None
    success = None
    def __init__(self):
        print("init tracker")
        self.instance = cv2.TrackerKCF_create()

    def update(self,frame):
        '''
        coords outputs the box parameter x,y,w,h tuple
        '''
        coords = None
        if self.initBB is not None:
            (H, W) = frame.shape[:2]
            (success, box) = self.instance.update(frame)
            self.success = success
            # If successful at tracking
            if success:
                (x,y,w,h) = [int(v) for v in box]
                coords = {'x':x,'y':y,'w':w,'h':h}
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
                info = [
                    ("Tracker", "KCF"),
                    ("Success", "Yes" if success else "No"),
                ]
                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return coords

    def check_success(self):
        '''Check if we get a succesful frame'''
        return self.success

    def setRoi(self,frame,ROI):
        # Delete current Tracker
        del self.instance
        # Create New Tracker
        self.instance = cv2.TrackerKCF_create()
        self.initBB = ROI
        # print(self.initBB)
        self.instance.init(frame,self.initBB)

    def choose_roi(self,frame,WindowName):
        # Select ROI
        #WindowName = "Select ROI"
        ROI = cv2.selectROI(WindowName, frame, fromCenter=False, showCrosshair=True)
        #cv2.destroyWindow(WindowNane)
        self.setRoi(frame,ROI)
        return frame
