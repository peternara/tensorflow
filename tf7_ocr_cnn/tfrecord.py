import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import os.path

IMAGE_DIR = './png/train/'
LABEL_FILE = './label.txt'
TRAIN_IMAGE_DIR = './sample'
TEST_IMAGE_DIR = './test'

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
    global  sample_list
    cnt = 0
    writer = tf.python_io.TFRecordWriter(output)
    for parent, dirnames, filenames in os.walk(input):
        for filename in filenames:
            cnt = cnt + 1
            fullname = os.path.join(parent, filename)
            label = parent[-1]
            #print(label)
            print(cnt)
            img_org = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(img_org, (32, 50), interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(img_org, (20, 20), interpolation=cv2.INTER_CUBIC)
            img_raw = img.tobytes()
            label_raw = np.zeros(54)
            # must astype, maybe size is not algined with conv to string size
            label_raw = label_raw.astype(np.uint8)
            label_raw[sample_list.index(label)] = 1
            #print(label_raw)
            label_raw = np.reshape(label_raw, [1, 54])
            label_raw = label_raw.tostring()  # 这里是把ｃ换了一种格式存储
            train = tf.train.Example(features=tf.train.Features(feature={
                'cnt': tf.train.Feature(int64_list=tf.train.Int64List(value=[cnt])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))
            }))
            writer.write(train.SerializeToString())  # 序列化为字符串
    writer.close()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        tfrecord(TRAIN_IMAGE_DIR, "train.tfrecords")
    elif sys.argv[1] == 'test':
        tfrecord(TEST_IMAGE_DIR, "test.tfrecords")
