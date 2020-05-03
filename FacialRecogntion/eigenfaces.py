#! /usr/bin/env python
############### IMPORTS ###############
import sys, os
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import ceil, sqrt
############### Paramters/Constants ###############

directory = '/home/kieren/FYP/MarmoTracker/libs/scrape_large_2_faces/'


############### Class ###############

class EigenFaces:
    def resize(self,image, min_rows, min_cols):
        '''
            Takes images in and resizes them as per the required number of rows and colums
        '''
        dim = (min_cols, min_rows) # (width, height)
        resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        return resized

    def to_Black_and_White(self,color):
        return cv.cvtColor(color,cv.COLOR_BGR2GRAY)

    def load_images(self,directory):
        ''' Loads images into array and returns them'''
        image_list = os.listdir(directory)
        original_images = []
        for image in image_list:
            img = cv.imread(directory+image)
            original_images.append(img)
        return original_images

    def plotFirst_N_Images(self,num=4):
        '''
        Args:
            images: List of images
            num: Number (N) images to plot
        Returns:
            None
        '''
        images=self.Singular_Images
        plt.figure("First {} components".format(num))
        cell_no = ceil(sqrt(num))
        for i in range(0,num):
            plt.subplot(cell_no,cell_no,i+1)
            plt.imshow(images[i,:,:],cmap="gray")

        plt.show()

    def image_pipeline(self,image_in):
        bandw = self.to_Black_and_White(image_in)
        small = self.resize(bandw,self.min_rows,self.min_cols)
        return small

    def __init__(self):
        original_images = self.load_images(directory)
        min_rows, min_cols = 32,32
        self.min_rows, self.min_cols = min_rows, min_cols
        new_images = list(map(self.image_pipeline,original_images))
        stacked_array = np.stack(new_images)
        two_dim = np.reshape(stacked_array,(-1,min_rows*min_cols))
        #Compute the mean:
        self.mean_image = np.mean(two_dim,axis=0)
        self.mean_subtracted = (two_dim - self.mean_image).transpose()
        #Apply PCA on Covariance Matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.mean_subtracted, full_matrices = False)
        self.VT = VT
        # self.Singular_Images = np.reshape(self.U,(self.U.shape[0],min_rows,min_rows))

    def load_image(self, path):
        '''Takes in path and returns image array'''
        return cv.imread(path)

    def project(self, image):
        vector_image = np.reshape(self.image_pipeline(image),(self.min_rows*self.min_cols))
        print("vec img shape {}".format(vector_image.shape))
        weights = np.dot(self.U.transpose(), vector_image - (self.mean_image).transpose())[3:20] # remove the first 3 eigenfaces to reduce lighting influence
        return weights

    def testno(self):
        print("VT shape")
        print(self.VT.shape)
        V = np.transpose(self.VT)
        print( "V shape")
        print(V.shape)
        print("mean_subtracted shape{}".format(self.mean_subtracted.shape))
        U = self.mean_subtracted.dot(V)
        print("u shape {}".format(U.shape))

    def euclidean_distance(self, weights1, weights2):
        ''' takes weights and returns euclidean distance '''
        return np.linalg.norm(weights1-weights2)