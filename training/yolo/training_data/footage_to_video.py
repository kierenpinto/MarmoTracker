#!/usr/bin/env python
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import time
import datetime

rs.device.as_playback
config = rs.config()
config.enable_device_from_file('./20200323_165013.bag')
pipe = rs.pipeline()
pipe.start(config)
frame_width = 1280
frame_height = 720
profile = pipe.get_active_profile()
stream = profile.get_streams()[0]
device = profile.get_device()
playback = device.as_playback()

framerate = stream.fps()
print("framerate {}".format(framerate))
duration = playback.get_duration().total_seconds()
print("duration {}".format(duration))
out = cv.VideoWriter('20200323_165013.avi',cv.VideoWriter_fourcc('M','J','P','G'), framerate, (frame_width,frame_height))
time.sleep(1)
while True:
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    #time.sleep(0.1)
    try:
        color_image = np.asanyarray(color_frame.get_data())
        img = cv.cvtColor(color_image,cv.COLOR_RGB2BGR)
        position = playback.get_position()/1000000000
        print("frame number: {} timestamp: {} position {}".format(frames.frame_number, frames.timestamp,position))
        out.write(img)
        cv.imshow('people_image',img)
        pass
    except Exception as e:
        print(e)
        pass
    if cv.waitKey(1) == ord('q'):
        break
    if position >= duration:
        break

out.release()
