#! /usr/bin/env python
import os
from fisherfaces import FisherFaces

instance = FisherFaces()

instance.add_class('./test_faces/1/')
instance.add_class('./test_faces/2/')

instance.load()
instance.train()
# print(instance.fisher_eigenvectors.shape)

weights1 = instance.project('./test_faces/1/551.jpg-1')
weights2 = instance.project('./test_faces/2/29.jpg-1')
weights_test = instance.project('./test_faces/2/19.jpg-1')

for test_img in os.listdir('./test_faces/2/'):
    weights_test = instance.project('./test_faces/2/'+test_img)
    dist1 = instance.euclidean_distance(weights_test,weights1)
    dist2 = instance.euclidean_distance(weights_test,weights2)
    if dist1<dist2:
        marmoset = 1
        # print("marmoset1 "+test_img)
    else:
        marmoset = 2
        # print("marmoset2 "+test_img)
    print("marmoset: {} image: {} dist1: {} dist2 {}".format(marmoset,test_img, dist1,dist2))


# instance.plotFirst_N_Images(20)