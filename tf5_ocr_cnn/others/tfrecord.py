import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import os.path


LABEL_FILE = 'label.txt'
TRAIN_IMAGE_DIR = './source/train/'
TEST_IMAGE_DIR = './source/test/'

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#IMAGE_HEIGHT = 114
#IMAGE_WIDTH = 450
IMAGE_HEIGHT = 25
IMAGE_WIDTH = 100

MAX_CAPTCHA = 6
CHAR_SET_LEN = 26

def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    vector = vector.astype(np.uint8)
    for i, c in enumerate(name):
        idx = i * 26 + ord(c) - 97
        vector[idx] = 1
    return vector


def tfrecord(input, output):
    cnt = 0
    writer = tf.python_io.TFRecordWriter(output)
    pnt_path = input
    for parent, dirnames, filenames in os.walk(pnt_path):
        for filename in filenames:
            print(cnt)
            # image
            fullname = os.path.join(parent, filename)
            img = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 25), interpolation=cv2.INTER_CUBIC)
            #height, width = img.shape
            #print(height, width)
            #img = cv2.resize(img_org, (100, 20), interpolation=cv2.INTER_CUBIC)
            img_raw = img.tobytes()
            #label
            print(filename[:-5])
            label = name2vec(filename[:-5])
            #print(label)
            label_raw = np.reshape(label, [1, MAX_CAPTCHA*CHAR_SET_LEN])
            label_raw = label_raw.tostring()  # 这里是把ｃ换了一种格式存储
            train = tf.train.Example(features=tf.train.Features(feature={
                'cnt': tf.train.Feature(int64_list=tf.train.Int64List(value=[cnt])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))
            }))
            print(img.shape)
            print(label.shape)
            writer.write(train.SerializeToString())  # 序列化为字符串'''
            cnt = cnt + 1
    writer.close()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        tfrecord(TRAIN_IMAGE_DIR, "train.tfrecords")
    elif sys.argv[1] == 'test':
        tfrecord(TEST_IMAGE_DIR, "test.tfrecords")
