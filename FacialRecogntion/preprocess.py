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


############### Functions ###############


def resize(image, min_rows, min_cols):
    '''
        Takes images in and resizes them as per the required number of rows and colums
    '''
    dim = (min_cols, min_rows) # (width, height)
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    return resized

def to_Black_and_White(color):
    return cv.cvtColor(color,cv.COLOR_BGR2GRAY)

def load_images(directory):
    ''' Loads images into array and returns them'''
    image_list = os.listdir(directory)
    original_images = []
    for image in image_list:
        img = cv.imread(directory+image)
        original_images.append(img)
    return original_images

def smallest_size(original_images):
    '''Find smallest common image size'''
    min_rows, min_cols = sys.maxsize, sys.maxsize
    max_rows, max_cols = 0, 0
    for (i, image) in enumerate(original_images):

        r, c = image.shape[0], image.shape[1]
        min_rows = min(min_rows, r)
        max_rows = max(max_rows, r)
        min_cols = min(min_cols, c)
        max_cols = max(max_cols, c)
        
    print("\n==> Least common image size:", min_rows, "x", min_cols, "pixels")
    return min_rows,min_cols

def view():
    # Test function to view images
    original_images = load_images(directory)
    min_rows,min_cols = smallest_size(original_images)
    for image in original_images:
        new_im = resize(image, min_rows, min_cols)
        cv.imshow('old image', image)
        cv.imshow('new image', new_im)
        while(True):
            key=chr(cv.waitKey(0))
            print(key)
            if key == 'q':
                return
            elif key == 'n':
                break
            else:
                print("Press q to quit and n for next")

def plotFirst_N_Images(images,num=4):
    '''
    Args:
        images: List of images
        num: Number (N) images to plot
    Returns:
        None
    '''
    plt.figure("First {} components".format(num))
    cell_no = ceil(sqrt(num))
    for i in range(0,num):
        plt.subplot(cell_no,cell_no,i+1)
        plt.imshow(images[i,:,:],cmap="gray")

def main():
    print("called as Main")
    original_images = load_images(directory)
    min_rows,min_cols = smallest_size(original_images)
    min_rows, min_cols = 128,128
    def image_pipeline(image_in):
        bandw = to_Black_and_White(image_in)
        small = resize(bandw,min_rows,min_cols)
        return small
    new_images = list(map(image_pipeline,original_images))
    stacked_array = np.stack(new_images)
    print("Stacked array shape:{}".format(stacked_array.shape))
    two_dim = np.reshape(stacked_array,(-1,min_rows*min_cols))
    print("2D matrix shape:{}".format(two_dim.shape))
    #Compute the mean:
    mean_image = np.mean(two_dim,axis=0)
    print("Mean image shape:{}".format(mean_image.shape))
    mean_subtracted = two_dim - mean_image
    print("Mean subtracted shape:{}".format(mean_subtracted.shape))
    #Apply PCA
    U, Sigma, VT = np.linalg.svd(mean_subtracted, full_matrices = False)
    # Sanity check on dimensions
    print("U:", U.shape)
    print("Sigma:", Sigma.shape)
    print("V^T:", VT.shape)
    # plt.figure("Singular Values")
    # plt.plot(np.arange(0,len(Sigma)),Sigma)
    # Reshape Right Singular Matrix (VT) back to images 199x28x28
    Singular_Images = np.reshape(VT,(VT.shape[0],min_rows,min_rows))
    plotFirst_N_Images(Singular_Images,20)
    # imgplt = plt.figure("IMG")
    # plt.imshow(Singular_Images[0,:,:])
    plt.show()

if __name__ == "__main__":
    main()