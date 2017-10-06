import cv2
import sys
import os
import numpy as np


if __name__ == '__main__':
    print('start conv...')
    #png_path = './train/png/'
    png_path = sys.argv[1]
    save_path = './conv/'
    noise = np.random.normal(10, 1.5, (20, 100))
    noise = noise.astype(np.uint8)
    for parent, dirnames, filenames in os.walk(png_path):
        for filename in filenames:
            fullname = os.path.join(parent, filename)
            img = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 20), interpolation=cv2.INTER_CUBIC)
            savename = os.path.join(save_path, filename)
            savename1 = savename[:-4] + '_1.png'
            savename2 = savename[:-4] + '_2.png'
            savename3 = savename[:-4] + '_3.png'
            
            img1 = img - noise
            img2 = img - 20
            #print(savename)
            cv2.imwrite(savename1, img) 
            cv2.imwrite(savename2, img1)
            cv2.imwrite(savename3, img2)

