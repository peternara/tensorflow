import cv2
import numpy as np
import os


NOISE_PATH = 'source/noise/'
SOURCE_PATH = 'source/org/train/png/'
FILTER_PATH = 'source/filter/'

if __name__ == '__main__':
    cnt = 0
    for parent, dirnames, filenames in os.walk(SOURCE_PATH):
        for filename in filenames:
            imgname = str(cnt + 1) + '.png'
            # image
            fullname = os.path.join(parent, imgname)
            print(fullname)
            img_org = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            img_filter = cv2.imread(FILTER_PATH+imgname, cv2.IMREAD_GRAYSCALE)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            img_filter = cv2.dilate(img_filter, kernel)
            retval, img_conv = cv2.threshold(img_filter, 100, 255, cv2.THRESH_BINARY_INV)
            img_noise = img_org & img_conv
            img_org[img_conv == 0] = 255
            cv2.imwrite(NOISE_PATH+imgname, img_org)
            cnt = cnt + 1