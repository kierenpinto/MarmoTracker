import pyrealsense2 as rs
import cv2 as cv
import numpy as np

class depth_camera:
    # face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    def __init__(self):
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,15)
        self.config.enable_stream(rs.stream.depth,640,480,rs.format.y16,15)
        self.pipe = rs.pipeline()
    def start(self):
        ''' Starts the camera '''
        self.pipe.start()

    def update_frame(self):
        ''' Gets the next frame '''
        self.frames = self.pipe.wait_for_frames()
        # return self

    def get_depth(self):
        depth_frame = self.frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image

    def to_grey(self,color_image):
        grayscale_image = cv.cvtColor(color_image,cv.COLOR_RGB2GRAY)
        return grayscale_image

    def get_color(self):
        color_frame = self.frames.get_color_frame()
        color_image_bgr = np.asanyarray(color_frame.get_data())
        color_image = cv.cvtColor(color_image_bgr,cv.COLOR_RGB2BGR)
        return color_image

    def get_area_depth_rect(self,depth_image,x1,y1,x2,y2):
        """
        Finds the average depth of an area specified by a rectangle. 
        """
        face_rectangle = depth_image[y1:y2,x1:x2]
        average = np.average(face_rectangle)
        return average/1000

    def getCentre(x,y,w,h):
        ''' input edges of rectangele '''
        x1,y1,x2,y2 = x,y,x+w,y-h
        return int(np.average([x1,x2]).round()),int(np.average([y1,y2]).round())

    def deproject_depth(self,depth_frame,x,y):
        depth = depth_frame.get_distance(x,y)
        x,y,z = rs.rs2_deproject_pixel_to_point(depth_stream_intrinsics,pixel,depth)
        return x,y,z

    def stop(self):
        self.pipe.stop()