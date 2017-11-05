import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import os.path

TRAIN_IMAGE_DIR = './train/'
TEST_IMAGE_DIR = './test/'

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

label_idx = 54
label_len = 6


def tfrecord(input, output):
    cnt = 0
    writer = tf.python_io.TFRecordWriter(output)

    for parent, dirnames, filenames in os.walk(input):
        for filename in filenames:
            # image
            fullname = os.path.join(parent, filename)
            img = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 20), interpolation=cv2.INTER_CUBIC)
            #ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
            img_raw = img.tobytes()

            #label
            label = filename[-(4+label_len):-4]
            print("%d: %s" %(cnt, label))
            label_raw = np.zeros(label_idx*label_len) #8 * 3
            label_raw = label_raw.astype(np.uint8)
            for i in range(label_len):
                #print(label[i])
                label_raw[sample_list.index(label[i]) + label_idx*i] = 1
            #print(label_raw)
            label_raw = np.reshape(label_raw, [1, label_idx*label_len])
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
