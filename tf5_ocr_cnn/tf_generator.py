import numpy as np
import os
import os.path
import sys
import cv2
import random

SAMPLE_DIR = './source/sample/'

'''sample_list = [
        '2', '3', '4', '5', '6', '7',
        '8', '9']'''

sample_list = [
        '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd',
        'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'm', 'n', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x',
        'y', 'z', 'A', 'B', 'C', 'D',
        'E', 'F', 'G', 'H', 'K', 'M',
        'N', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z']

NOISE_NUM = 586
NOISE_DIR = './source/noise/'

sample_len = 6
sample_idx = 54

def random_label_generator():
    list = []
    for i in range(0, sample_len):
        list.append(sample_list[random.randint(0, sample_idx-1)])
    print(list)
    return list

#single img 35*50 (5gap - 25image - 5gap)
#full img 200*50

def random_image_generator(idx, label, path):
    total_width = 200
    start = 0
    total_img = np.zeros((50,200), np.uint8)
    width_list = []
    path_list = []
    sum = 0
    for i in range(len(label)):
        path_list.append(get_single_image_path(label[i]))
        img = cv2.imread(path_list[i], cv2.IMREAD_GRAYSCALE)
        sum = sum + img.shape[1]
        width_list.append(img.shape[1])

    sum = sum - 10*len(label)
    if sum > 200:
        #print("Over 200 size, ignore")
        return None
    '''else:
        print('width list:')
        print(width_list)
        print('sum is %d' %sum)'''


    for i in range(len(label)):
        #print('i: %d' %i)
        img = cv2.imread(path_list[i], cv2.IMREAD_GRAYSCALE)
        height = img.shape[0]
        width = img.shape[1]
        #print('width: %d' %width)
        roiImg = img[:, 5:width-5]
        start = start + random.randint(0, total_width - sum)
        sum = sum - width + 10
        #print('%s: %d-%d' %(label[i], start, start+width-10))
        delta = random.uniform(0.3, 0.7)
        roiImg = delta * roiImg
        total_img[:, start:start+width-10] = roiImg
        total_width = 200 - start - width + 10
        start = start + width - 10

    #gassuian noise
    noise1 = np.random.normal(1, 0.2, (50, 200))

    #salt and papper noise
    noise2 = np.ones((50, 200), dtype=int)
    for i in range(10):
        noise2 = noise2 * np.random.randint(0, 2, (50, 200))
    noise = noise2 * 250
    noise = noise * (total_img == 0).astype(int)

    total_img = noise1 * total_img + noise
    total_img.astype(int)


    #org noise
    noise_idx = random.randint(1, NOISE_NUM)
    noise_path = NOISE_DIR + str(noise_idx) + '.png'
    img_noise = cv2.imread(noise_path, cv2.IMREAD_GRAYSCALE)
    img_noise = 255 -img_noise

    total_img = total_img + img_noise
    total_img = 255 - total_img
    cv2.imwrite(path + '/' + str(idx) + '_' + ''.join(label) + '.png', total_img)

def get_single_image_path(label):
    image_path = SAMPLE_DIR + label
    for parent, dirnames, filenames in os.walk(image_path):
        idx = random.randint(0, len(filenames)-1)
        path = image_path + '/' + filenames[idx]
        return path

if __name__ == "__main__":
    path = sys.argv[1]
    cnt = sys.argv[2]

    for i in range(0, int(cnt)):
        random_image_generator(i, random_label_generator(), path)