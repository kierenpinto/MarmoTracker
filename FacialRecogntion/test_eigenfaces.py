#! /usr/bin/env python

from eigenfaces import EigenFaces

instance = EigenFaces()

marmoset1 = instance.load_image('./test_faces/1/551.jpg-1')
marmoset2 = instance.load_image('./test_faces/2/29.jpg-1')


marm_test = instance.load_image('./test_faces/2/19.jpg-1')
# marm_test = instance.load_image('./test_faces/2/29.jpg-2')

weights1 = instance.project(marmoset1)
weights2 = instance.project(marmoset2)
weights_test = instance.project(marm_test)

dist1 = instance.euclidean_distance(weights_test,weights1)
dist2 = instance.euclidean_distance(weights_test,weights2)

if dist1<dist2:
    print("marmoset1")
else:
    print("marmoset2")

instance.testno()

# instance.plotFirst_N_Images(20)