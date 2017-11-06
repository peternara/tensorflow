import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import os.path


LABEL_FILE = 'label.txt'
PNG_DIR = 'png'
TRAIN_IMAGE_DIR = './source/org/train/'
TEST_IMAGE_DIR = './source/org/test/'
GENERATOR_IMAGE_DIR = './source/generator/'
NOISE_ENABLE = 0
label_idx = 54
label_len = 6
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 20

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
    png_path = input + PNG_DIR
    print(png_path)
    label_path = input + LABEL_FILE
    f_label = open(label_path, 'r')
    l_label = f_label.readlines()
    for parent, dirnames, filenames in os.walk(png_path):
        for filename in filenames:
            imgname = str(cnt+1) + '.png'
            # image
            fullname = os.path.join(parent, imgname)
            img_org = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img_org, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
            img_raw = img.tobytes()

            #label
            label = l_label[cnt][-7:-1]
            print("%d: %s" % (cnt, label))
            label_raw = np.zeros(IMAGE_WIDTH*IMAGE_HEIGHT) #54 * 6
            label_raw = label_raw.astype(np.uint8)
            for i in range(label_len):
                #print(label[i])
                label_raw[sample_list.index(label[i]) + label_idx*i] = 1
            #print(label_raw)
            label_raw = np.reshape(label_raw, [1, IMAGE_WIDTH*IMAGE_HEIGHT])
            label_raw = label_raw.tostring()  # 这里是把ｃ换了一种格式存储
            train = tf.train.Example(features=tf.train.Features(feature={
                'cnt': tf.train.Feature(int64_list=tf.train.Int64List(value=[cnt])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))
            }))
            writer.write(train.SerializeToString())  # 序列化为字符串'''
            cnt = cnt + 1
    writer.close()

    if output == "train.tfrecords" and NOISE_ENABLE == 1:
        writer = tf.python_io.TFRecordWriter('generator.tfrecords')
        cnt = 0
        for parent, dirnames, filenames in os.walk(GENERATOR_IMAGE_DIR):
            for filename in filenames:
                # image
                fullname = os.path.join(parent, filename)
                img = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
                # ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
                img_raw = img.tobytes()

                # label
                label = filename[-(4 + label_len):-4]
                print("%d: %s" % (cnt, label))
                label_raw = np.zeros(label_idx * label_len)  # 8 * 3
                label_raw = label_raw.astype(np.uint8)
                for i in range(label_len):
                    # print(label[i])
                    label_raw[sample_list.index(label[i]) + label_idx * i] = 1
                # print(label_raw)
                label_raw = np.reshape(label_raw, [1, label_idx * label_len])
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
