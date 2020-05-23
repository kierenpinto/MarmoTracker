
import numpy as np
import numpy.matlib
''' Convert from box coordinates to centre point '''

def boxToCentre(box):
    x,y,w,h = box
    cx = x + w/2
    cy = y + h/2
    return (cx,cy)

class MarmFaceMap:
    def __init__(self,marmoset,face):
        self.marmoset = marmoset
        self.face = face

def euclideanDistance(pointA, pointB):
    return numpy.linalg.norm(pointA-pointB)

def mapMarmosetToFace(faces,marmosets):
    num_faces = len(faces)
    num_marmosets = len(marmosets)
    if (num_faces < 1 or num_marmosets < 1):
        return

    marmoset_array = np.expand_dims(np.array(marmosets),axis=1)
    # print(marmoset_array)
    faces_array = np.array(faces)
    # print(faces_array)
    marmoset_matrix = np.transpose(np.repeat(marmoset_array,num_faces,axis=1),axes=(1,0,2))# Matrix of marmoset coordinates
    faces_matrix = np.tile(faces_array,num_marmosets).reshape((num_faces,num_marmosets,2))
    # print("marmoset matrix {}".format(marmoset_matrix))
    # print("faces matrix {}".format(faces_matrix))
    sub = marmoset_matrix - faces_matrix
    # print("sub: {}".format(sub))
    distances = np.linalg.norm(sub,axis=2)
    # print("distances: {}".format(distances))

    marmoset_close_to_face = np.argmin(distances,axis=1) # finds the closest marmoset to each face
    print(marmoset_close_to_face)
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

    '''
    if num_marmosets > num_faces:
        # more marmosets than faces
        marmoset_close_to_face = np.argmin(distances,axis=1) # finds the closest marmoset to each face
        marm_to_faces = [None] * num_marmosets
        for marmoset,face in enumerate(marmoset_close_to_face):
            marm_to_faces[marmoset] = face
        return marm_to_faces
    else: #more faces detected than marmosets 
        face_close_to_marmoset = np.argmin(distances,axis=0) # find the closest face to each marmoset
        return face_close_to_marmoset # face_num = marmosets_to_faces[marm_num]
    '''    
