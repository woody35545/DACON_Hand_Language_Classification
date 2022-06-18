import os 
import numpy as np
import torch
import cv2

# Seed 고정
import random
import numpy as np
import pandas as pd

from glob import glob

label_df = pd.read_csv('data/train.csv')
label_df.head()

label_df['label'][label_df['label'] == '10-1'] = 10 ## label : 10-1 -> 10
label_df['label'][label_df['label'] == '10-2'] = 0 ## Label : 10-2 -> 0
label_df['label'] = label_df['label'].apply(lambda x : int(x)) ## Dtype : object -> int

def get_train_data(data_dir):
    img_path_list = []
    label_list = []
    
    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.png')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    
    # get label
    label_list.extend(label_df['label'])
                
    return img_path_list, label_list

if __name__ == '__main__':

    all_img_path, all_label = get_train_data('data/train')

    for path in all_img_path:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = 0
        lab[:,:,0] = 0
        img2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        img3 = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        
        normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        normalized_hsv = cv2.normalize(hsv, None, 0, 255, cv2.NORM_MINMAX)
        normalized_lab = cv2.normalize(lab, None, 0, 255, cv2.NORM_MINMAX)
        normalized_img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        normalized_img3 = cv2.normalize(img3, None, 0, 255, cv2.NORM_MINMAX)

        laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        normalized_laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

        cv2.imshow("hsv", hsv)
        cv2.imshow("img", img)
        cv2.imshow("yuv", yuv)
        cv2.imshow("lab", lab)
        cv2.imshow("img2", img2)
        cv2.imshow("img3", img3)
        cv2.imshow("laplacian", laplacian)

        # cv2.imshow("normalized_img", normalized_img)
        cv2.imshow("normalized_hsv", normalized_hsv)
        cv2.imshow("normalized_lab", normalized_lab)
        cv2.imshow("normalized_img2", normalized_img2)
        cv2.imshow("normalized_img3", normalized_img3)
        cv2.imshow("normalized_laplacian", normalized_laplacian)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break