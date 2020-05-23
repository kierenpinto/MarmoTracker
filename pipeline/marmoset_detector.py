#!/usr/bin/python
import numpy as np
import time
import cv2
'''
Refer to https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py
https://github.com/opencv/opencv/issues/14414 
https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-5/

'''

class YOLODetector:
    directory = '/home/kieren/FYP/MarmoTracker/marmoset/YOLO/'
    configPath = directory+ 'yolov3-tiny-marmoset.cfg'
    weightsPath = directory+ 'yolov3-tiny-marmoset.backup'
    confidence_threshold = 0.05 #confidence threshold
    dimensions = (608,608)
    def __init__(self):
        net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.net = net
        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        output_layers = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        self.output_layers = output_layers

    def predict(self,image):
        (H, W) = image.shape[:2]
        img_sq = max(W,H)
        canvas = np.full((img_sq,img_sq,3),128, dtype=np.uint8) # create a 3D matrix

        canvas[(img_sq-H)//2:(img_sq-H)//2+H,\
            (img_sq-W)//2:(img_sq-W)//2+W,\
                :] = image # Add letterboxing

        image_blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, size=self.dimensions, swapRB=True, ddepth=cv2.CV_32F)
        
        self.net.setInput(image_blob)
        # start_time = time.time()
        layerOutputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        box_centres = []

        for output in layerOutputs:
            # loop over each region
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores) #maximum class
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence_threshold:
                    centerX, centerY = (img_sq*detection[0:2] -(img_sq-np.array([W, H]))//2).astype("int")
                    width, height = (img_sq*detection[2:4]).astype('int')
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    box_centres.append((centerX,centerY))
 
        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        return idxs,boxes,confidences,box_centres
    def predict_nms(self,image):
        idxs,boxes,confidences,box_centres = self.predict(image)
        nms_boxes = []
        nms_confidences = []
        nms_box_centres = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                nms_boxes.append(boxes[i])
                nms_confidences.append(confidences[i])
                nms_box_centres.append(box_centres[i])
        return nms_boxes, nms_confidences, nms_box_centres

    def draw_boxes(self,color_image_ptr,detected_marmosets):
        for (x,y,w,h) in detected_marmosets:
            cv2.rectangle(color_image_ptr,(x,y),(x+w,y+h),(0,255,0),2)

    def draw_boxes_nms(self,image,idxs,boxes,confidences):
        if len(idxs) >0 :
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = (255, 0, 0)
                print(x,y,w,h,confidences[i])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)