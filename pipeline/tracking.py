import numpy as np
from hungarian import Hungarian
from time import time as currentTime
from marmoset_detector import MarmosetBoxes
from position_tools import hungarianAlg, mapMarmosetToFace
from typing import TypeVar, Type
from face_recognition import FaceRecognition
from camera import depth_camera

# MarmosetBoxType = TypeVar('MarmosetBoxType',bound=MarmosetBoxes)

class Tracker:
    def __init__(self,cam:depth_camera):
        self.cacheBox = []
        self.sequenceCache = []
        self.sequences = []
        self.cam = cam
    
    def _prepBoxPoint(self,marmoBoxes:MarmosetBoxes,index):
        centrePoints = marmoBoxes.centres[index]
        xyz_points = self.cam.deproject(centrePoints).flatten()
        return centrePoints,xyz_points


    def updateSequence(self,marmBoxes:MarmosetBoxes):
        # if self.cacheBox is None: #First Run
        #     self.cacheBox = np.copy(marmBoxes.boxes) # copy and use as history
        #     # create new sequences
        #     for marm in marmBoxes.centres:
        #         seqId = len(self.sequences)+1
        #         seq = Sequence(seqId).addBoxPoint(marm)
        #         # seq = Sequence(seqId).addBoxPoint(*self._prepBoxPoint(marmBoxes,))
        #         self.sequences.append(seq)
        #         self.sequenceCache.append(seq)
        #     return
        # All successive updates start from here
        cur_to_prev_mappings =  [None] * marmBoxes.no_boxes
        # print(self.cacheBox,marmBoxes.boxes)
        hung_out = hungarianAlg(self.cacheBox,marmBoxes.boxes)
        for result in hung_out:
            current,previous = result
            cur_to_prev_mappings[current] = previous
        
        new_sequenceCache = []

        for currentBoxInd,previousBoxInd in enumerate(cur_to_prev_mappings): #current box index, previous box index
            
            if (previousBoxInd is not None): # there is a valid previous box in sequence
                #link
                seq = self.sequenceCache[previousBoxInd]
                #add data point to sequence
                # seq.addBoxPoint(marmBoxes.centres[currentBoxInd])
                seq.addBoxPoint(*self._prepBoxPoint(marmBoxes,currentBoxInd))
                #add to sequence cache
                new_sequenceCache.append(seq)

            else: #there is no valid previous box in sequence
                #create new sequence
                seqId = len(self.sequences)+1
                seq = Sequence(seqId)
                # seq.addBoxPoint(marmBoxes.centres[currentBoxInd])
                seq.addBoxPoint(*self._prepBoxPoint(marmBoxes,currentBoxInd))
                self.sequences.append(seq)
                new_sequenceCache.append(seq)

        self.sequenceCache = new_sequenceCache # set a new sequence cache at end of update
        self.cacheBox = np.copy(marmBoxes.boxes)
        return

    def updateID(self,faceBoxes,faceIds,marmBoxes):
        marm_to_face = mapMarmosetToFace(faceBoxes.centres,marmBoxes.centres)
        marmosetFrameLabels = [None] * len(marm_to_face)
        for marmoset,face in enumerate(marm_to_face):
            if face is not None:
                self.sequenceCache[marmoset].setMarmoset(faceIds[face])
                marmosetFrameLabels[marmoset] = faceIds[face]
        return marmosetFrameLabels

    def update(self,faceBoxes,faceIds,marmBoxes):
        self.updateSequence(marmBoxes)
        return self.updateID(faceBoxes,faceIds,marmBoxes)
        
    def getPlotPoints(self):
        #Finish THIS
        if len(self.sequenceCache)>0:
            return self.sequenceCache[0]
        else:
            return None
        

class Sequence:
    ''' tracks a sequence of movements made by a single marmoset '''
    arr = []

    def __init__(self,id):
        self.id = id

    def setMarmoset(self,marmo_id):
        self.marmoset = marmo_id
        return self

    def addBoxPoint(self,pixel,xyz):
        point = (currentTime(), pixel, xyz) #t,x,y,z
        self.arr.append(point)
        return self
