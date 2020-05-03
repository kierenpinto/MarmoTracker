#! /usr/bin/env python
#Fisher Faces
import sys, os
import numpy as np 
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import ceil, sqrt

class FisherFaces:
    directory_list = []
    number_of_classes = 0
    min_rows, min_cols = 32,32
    num_components = 20 # this is the number of eigenvectors used to reconstruct (ie principle components)
    def add_class(self,directory):
        self.number_of_classes = self.number_of_classes + 1
        self.directory_list.append(directory)

    def resize(self,image, min_rows, min_cols):
        '''
            Takes images in and resizes them as per the required number of rows and colums
        '''
        dim = (min_cols, min_rows) # (width, height)
        resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        return resized

    def to_Black_and_White(self,color):
        return cv.cvtColor(color,cv.COLOR_BGR2GRAY)

    def image_pipeline(self,image_in):
        bandw = self.to_Black_and_White(image_in)
        small = self.resize(bandw,self.min_rows,self.min_cols)
        return small

    def load(self):
        classes_list = []
        for directory in self.directory_list:
            image_list = os.listdir(directory)
            original_images = []
            for image in image_list:
                img = cv.imread(directory+image)
                original_images.append(img)
            resized_images = list(map(self.image_pipeline,original_images))
            stacked_images = np.stack(resized_images)
            stacked_image_vectors = np.reshape(stacked_images,(-1,self.min_rows*self.min_cols))
            classes_list.append(stacked_image_vectors)
        # classes_stack = np.stack(classes_list)
        self.classes_list = classes_list
        # self.classes_stack = classes_stack

    def lda(self):
        ''' Linear Discriminant analysis '''
        X = self.classes_list
        # print("classes stack shape {}".format(X[0].shape))
        mu_i = np.stack(list(map(lambda xi: np.mean(xi,axis=0),X)), axis=0)
        mu = np.mean(mu_i, axis=0)
        # print("classes mean shape {} and {}".format(mu_i.shape,mu_i[1].shape))
        # print("mean shape {}".format(mu.shape))
        img_dim = X[0].shape[1]
        Sb = np.zeros((img_dim,img_dim), dtype=np.float32)
        Sw = np.zeros((img_dim,img_dim), dtype=np.float32)
        for i in range(len(X)):
            n = X[i].shape[0]
            Sb = Sb + np.dot((mu_i[i]-mu).T, (mu_i[i]-mu) ) * n
            Sw = Sw + np.dot((X[i]-mu_i[i]).T, (X[i]-mu_i[i]))
        eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(Sw)*Sb)
        idx = np.argsort(-eigenvals.real) # find the maximum eigenval argument
        eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:,idx] #
        self.eigenvals_lda = np.array(eigenvals[0:self.num_components].real, dtype=np.float32, copy=True)
        self.eigenvecs_lda = np.array(eigenvecs[0:self.num_components].real, dtype=np.float32, copy=True)

    def pca(self):
        ''' Principle component analysis '''
        X = np.vstack(self.classes_list) # Join both classes together (concatenate) for PCA
        # print("classes stack shape {}".format(X.shape))
        mean_imgs = np.mean(X, axis=0)
        self.mean_image_pca = mean_imgs
        mean_sub = (X - mean_imgs).transpose()
        # apply PCA on covariance matrix
        U, sigma, VT = np.linalg.svd(mean_sub, full_matrices=False)
        UT = U[:,0:self.num_components]
        # print("U shape {}".format(UT.shape))
        self.eigenvecs_pca = UT
        self.eigenvals_pca = sigma

    def euclidean_distance(self, weights1, weights2):
        ''' takes weights and returns euclidean distance '''
        return np.linalg.norm(weights1-weights2)

    def project(self, image_path):
        ''' Project Image into vector space '''
        image = cv.imread(image_path)
        vector_image = np.reshape(self.image_pipeline(image),(self.min_rows*self.min_cols))
        # print("vec img shape {}".format(vector_image.shape))
        weights = np.dot(self.fisher_eigenvectors, vector_image - (self.mean_image_pca).transpose())[3:] # remove the first 3 eigenfaces to reduce lighting influence
        return weights

    def plotFirst_N_Images(self,num=4):
        '''
        Args:
            images: List of images
            num: Number (N) images to plot
        Returns:
            None
        '''
        images=self.fisher_eigenvectors
        images = np.reshape(images,(images.shape[0],self.min_rows,self.min_rows))
        plt.figure("First {} components".format(num))
        cell_no = ceil(sqrt(num))
        for i in range(0,num):
            plt.subplot(cell_no,cell_no,i+1)
            plt.imshow(images[i,:,:],cmap="gray")

        plt.show()
    
    def train(self):
        ''' Run training '''
        self.lda()
        print("finished lda")
        self.pca()
        print("finished pca")
        self.fisher_eigenvectors = np.dot(self.eigenvecs_pca, self.eigenvecs_lda) #Final W in the formula
