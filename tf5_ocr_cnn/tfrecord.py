import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import os.path


LABEL_FILE = 'label.txt'
PNG_DIR = 'png'
TRAIN_IMAGE_DIR = './source/train/'
TEST_IMAGE_DIR = './source/test/'

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


def tfrecord(input, output):
    cnt = 0
    writer = tf.python_io.TFRecordWriter(output)
    pnt_path = input + PNG_DIR
    label_path = input + LABEL_FILE
    f_label = open(label_path, 'r')
    l_label = f_label.readlines()
    for parent, dirnames, filenames in os.walk(pnt_path):
        for filename in filenames:
            print(cnt)
            # image
            fullname = os.path.join(parent, filename)
            img_org = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img_org, (100, 20), interpolation=cv2.INTER_CUBIC)
            img_raw = img.tobytes()

            #label
            label = l_label[cnt][-7:-1]
            label_raw = np.zeros(324) #54 * 6
            label_raw = label_raw.astype(np.uint8)
            for i in range(6):
                #print(label[i])
                label_raw[sample_list.index(label[i]) + 54*i] = 1
            #print(label_raw)
            label_raw = np.reshape(label_raw, [1, 324])
            label_raw = label_raw.tostring()  # 这里是把ｃ换了一种格式存储
            train = tf.train.Example(features=tf.train.Features(feature={
                'cnt': tf.train.Feature(int64_list=tf.train.Int64List(value=[cnt])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))
            }))
            writer.write(train.SerializeToString())  # 序列化为字符串'''
            cnt = cnt + 1
    writer.close()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        tfrecord(TRAIN_IMAGE_DIR, "train.tfrecords")
    elif sys.argv[1] == 'test':
        tfrecord(TEST_IMAGE_DIR, "test.tfrecords")
