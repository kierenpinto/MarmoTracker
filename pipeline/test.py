from position_tools import mapMarmosetToFace,percentOverlap,boxToCentre,hungarianAlg
from hungarian import Hungarian
from face_recognition import FaceRecognition
import cv2 as cv
import numpy as np
import time
def testMapMarmosetToFace():
    marmosets =[(790.0, 451.0), (164.5, 563.5), (34.5, 615.5), (67.5, 659.5)]
    faces = [(839, 357),(837, 355)]
    marmo_map = mapMarmosetToFace(faces,marmosets)
    print(marmo_map)

def testMarmoPosTrack():
    prev = np.array([[1,2,3,4],[3,2,4,1]])
    # cur = np.array([[4,3,2,1],[1,5,4,2]])
    cur = np.array([[1,5,4,2]])
    print(hungarianAlg([],cur))
    # print(percentOverlap([1,2,3,4],[1,5,4,2]))

def testBoxToCentre():
    boxes = np.array([])
    print(boxToCentre(boxes))

def testHungarian():
    hung = [[38,53,61,36,66],[100,60,9,79,34],[30,37,36,72,24],[61,95,21,14,64],[89,90,4,5,79]]
    hungarian = Hungarian(hung, is_profit_matrix=False) #profit is false since we are minimizing cost
    hungarian.calculate()
    print(hungarian.get_results())


def testFaceRecogntion():
    fr = FaceRecognition()
    img = cv.imread('./test_faces/2/34.jpg-1')
    print(fr.recognize(img))

def testMultiFaceRecognition():
    fr = FaceRecognition()
    imgs = []
    imgs.append(cv.imread('./test_faces/2/34.jpg-1'))
    imgs.append(cv.imread('./test_faces/1/561.jpg-1'))
    start = time.time()
    print(fr.multiRecognise(imgs))
    print(time.time()-start)

if __name__ == "__main__":
    # testMapMarmosetToFace()
    # testBoxToCentre()
    # testFaceRecogntion()
    testMarmoPosTrack()
    # testMultiFaceRecognition()