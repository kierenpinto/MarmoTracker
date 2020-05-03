import cv2
import numpy as np
import os

def store_raw_images(pic_num = 1):
    if not os.path.exists('neg'):
        os.makedirs('neg')

    photo_list = os.listdir('./orig_neg')
    for file in photo_list:
        file_name = 'orig_neg/' + file
        print(file_name)
        try:
            img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (100,100))
            cv2.imwrite("neg/" + str(file), resized_image)
        except Exception as e:
            print (str(e))

store_raw_images(1)