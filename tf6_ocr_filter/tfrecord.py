import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import os.path


TRAIN_IMAGE_DIR = './source/org/train/png/'
TEST_IMAGE_DIR = './source/org/test/'
LABEL_IMAGE_DIR = './source/filter/'
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 20

def tfrecord(input, output):
    cnt = 0
    writer = tf.python_io.TFRecordWriter(output)
    for parent, dirnames, filenames in os.walk(input):
        for filename in filenames:
            imgname = str(cnt+1) + '.png'
            # image
            fullname = os.path.join(parent, imgname)
            print(fullname)
            img_org = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img_org, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
            img_in = img.tobytes()

            outname = os.path.join(LABEL_IMAGE_DIR, imgname)
            print(outname)
            img_filt = cv2.imread(outname, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.resize(img_filt, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
            retval, img2 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)
            img_out = img2.tobytes()

            train = tf.train.Example(features=tf.train.Features(feature={
                'cnt': tf.train.Feature(int64_list=tf.train.Int64List(value=[cnt])),
                'img_in': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_in])),
                'img_out': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_out]))
            }))
            writer.write(train.SerializeToString())  # 序列化为字符串'''
            cnt = cnt + 1
    writer.close()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        tfrecord(TRAIN_IMAGE_DIR, "train.tfrecords")
    elif sys.argv[1] == 'test':
        tfrecord(TEST_IMAGE_DIR, "test.tfrecords")
