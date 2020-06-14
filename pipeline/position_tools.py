
import numpy as np
import numpy.matlib
from hungarian import Hungarian
''' Convert from box coordinates to centre point '''

def boxToCentre(box):
    if (box is not None):
        box = np.asarray(box)
        if(len(box.shape) == 1):
            box = np.expand_dims(box,axis=0)
        if (box.shape[1] == 4):
            x,y,w,h = box[:,0],box[:,1],box[:,2],box[:,3]
            cx = x + w/2
            cy = y + h/2
            return np.stack((cx,cy),axis=1)
        else:
            return []
    else:
        return []

def mapMarmosetToFace(faces,marmosets):
    debug = False
    num_faces = len(faces)
    num_marmosets = len(marmosets)
    if (num_faces < 1 or num_marmosets < 1):
        return []

    marmoset_array = np.expand_dims(np.array(marmosets),axis=1)
    print(marmoset_array) if debug else None
    faces_array = np.array(faces)
    print(faces_array) if debug else None
    marmoset_matrix = np.transpose(np.repeat(marmoset_array,num_faces,axis=1),axes=(1,0,2))# Matrix of marmoset coordinates
    faces_matrix = np.tile(faces_array,num_marmosets).reshape((num_faces,num_marmosets,2))
    print("marmoset matrix {}".format(marmoset_matrix)) if debug else None
    print("faces matrix {}".format(faces_matrix)) if debug else None
    sub = marmoset_matrix - faces_matrix
    print("sub: {}".format(sub)) if debug else None
    distances = np.linalg.norm(sub,axis=2)
    print("distances: {}".format(distances)) if debug else None

    marmoset_close_to_face = np.argmin(distances,axis=1) # finds the closest marmoset to each face
    print(marmoset_close_to_face) if debug else None
    face_close_to_marmoset = np.argmin(distances,axis=0) # find the closest face to each marmoset
    marm_to_faces = [None] * num_marmosets
    if num_marmosets > num_faces:
        for marmoset, face in enumerate(marmoset_close_to_face):
            if (face_close_to_marmoset[marmoset]==face):
                marm_to_faces[marmoset] = face
    else:
        for face, marmoset in enumerate(face_close_to_marmoset):
            if (marmoset_close_to_face[face] == marmoset):
                marm_to_faces[marmoset] = face

    return marm_to_faces  

def recXYWHtoXYXY(rectangle):
    x1,y1,w,h = rectangle
    x2, y2 = x1 + w, y1 + h
    return x1,y1,x2, y2

def intersectionOverUnion(rectangle1,rectangle2):
    ''' 
    Calculates the intersection over union of two rectangles
    Ref: https://www.geeksforgeeks.org/total-area-two-overlapping-rectangles/ 
    '''
    rectangle1 = recXYWHtoXYXY(rectangle1)
    rectangle2 = recXYWHtoXYXY(rectangle2)
    l1x,l1y, r1x,r1y = rectangle1
    l2x,l2y, r2x,r2y = rectangle2
    area = (min(r1x,r2x) - max(l1x,l2x)) * (min(r1y,r2y) - max(l1y,l2y))
    return area

def percentOverlap(previousBox,currentBox):
    areaP = previousBox[2] * previousBox[3]
    areaC = currentBox[2] * currentBox[3]
    areaI = intersectionOverUnion(previousBox,currentBox)
    totalArea = areaP + areaC - areaI
    return areaI/totalArea

def findOverPrev(lastBoxCache,currentBoxList,overlap_thres):
    '''
        Finds the overlap between faces in current frame with those from previous frames
    '''
    debug = False
    new_cache = [] # Cache is structured [[x,y,w,h,count]]
    for currentBox in currentBoxList:
        thres = overlap_thres
        new_box = None
        for lastBox in lastBoxCache:
            overlap = percentOverlap(lastBox[0:4],currentBox)
            print("Overlap {}".format(overlap)) if debug else None
            if overlap > thres:
                if len(lastBox)>4:
                    count = lastBox[4]+1
                else:
                    count = 1
                thres = overlap
                new_box = np.append(currentBox,np.array([count]))
        if new_box is None:
            new_box =  np.append(currentBox,np.array([0]))
            print("newbox {}".format(new_box)) if debug else None
        new_cache.append(new_box)
    return new_cache

def filteredBoxes(boxList,threshold):
    ''' Checks if face is stable in the time domain'''
    func = lambda box: box[4] > threshold
    box_with_count = filter(func, boxList)
    return list(map(lambda box: box[:4],box_with_count))

def vectorisedIOU(previousMarmosets,currentMarmosets):
    num_prev = len(previousMarmosets)
    num_cur = len(currentMarmosets)
    if num_cur==0 or num_prev==0:
        return []
    prev_array = np.expand_dims(np.array(previousMarmosets),axis=0)
    curr_array = np.expand_dims(np.array(currentMarmosets),axis=1)
    prev_matrix = np.repeat(prev_array,num_cur,axis=0)
    curr_matrix = np.repeat(curr_array,num_prev,axis=1)
    prev = np.concatenate((prev_matrix[:,:,0:2],prev_matrix[:,:,0:2] + prev_matrix[:,:,2:]),axis=2) #convert from x,y,w,h to x1,y1,x2,y2
    curr = np.concatenate((curr_matrix[:,:,0:2],curr_matrix[:,:,0:2] + curr_matrix[:,:,2:]),axis=2)
    areaI = np.maximum((np.minimum(prev[:,:,2],curr[:,:,2])-np.maximum(prev[:,:,0],curr[:,:,0]))*(np.minimum(prev[:,:,3],curr[:,:,3])-np.maximum(prev[:,:,1],curr[:,:,1])),0)
    areaP = prev_matrix[:,:,2] * prev_matrix[:,:,3] # width x height
    areaC = curr_matrix[:,:,2] * curr_matrix[:,:,3]
    areaT = areaP+areaC-areaI
    IOU = areaI*100/areaT #intersection over union percentage overlap
    return IOU


def hungarianAlg(previous,current):
    IOU = vectorisedIOU(previous,current)
    # print(IOU)
    if len(IOU) == 0:
        return []
    hungarian = Hungarian(IOU, is_profit_matrix=True) #profit is true because we are maximizing cost matrix
    hungarian.calculate()
    return hungarian.get_results()