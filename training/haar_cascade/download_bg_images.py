import csv
import cv2
import urllib.request
import numpy as np
import os

# To run on massive m3 cluster: module load python/3.5.2-gcc4

# The name of the csv file to be used
filename = 'random_image_urls.csv'
# Creates a directory for the original negative images to be put in.
if not os.path.exists('orig_neg'):
    os.makedirs('orig_neg')

pic_num = 1
with open(filename, newline='') as csvfile:
    # open csv file
    image_reader = csv.reader(csvfile, delimiter=' ', quotechar='"')
    # iterate through each of the 
    for image in image_reader:
        # print(', '.join(image))
        print(image[0])
        try:
            i = image[0]
            urllib.request.urlretrieve(i, "orig_neg/" +  str(pic_num) + '.jpg')
            pic_num += 1
        except Exception as e:
            print(e)

