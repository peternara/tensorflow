import tensorflow as tf
import numpy as np
import sys
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_WIDTH = 100 #200
IMAGE_HEIGHT = 20 #50
batch_size = 64
LOG_PATH = './log/'

def tf_ocr_train(train_method, train_step, result_process, method='train'):
    global predict

    def read_and_decode(tf_record_path):  # read iris_contact.tfrecords
        filename_queue = tf.train.string_input_producer([tf_record_path])  # create a queue

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # return file_name and file
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'cnt': tf.FixedLenFeature([], tf.int64),
                                               'img_in': tf.FixedLenFeature([], tf.string),
                                               'img_out': tf.FixedLenFeature([], tf.string),
                                           })  # return image and label

        img_in = tf.decode_raw(features['img_in'], tf.uint8)
        img_in = tf.reshape(img_in, [2000])  # reshape image
        img_in = tf.cast(img_in, tf.float32) * (1. / 255)

        img_out = tf.decode_raw(features['img_out'], tf.uint8)
        img_out = tf.reshape(img_out, [2000])  # reshape image
        img_out = tf.cast(img_out, tf.float32) * (1. / 255)

        cnt = features['cnt']  # throw label tensor
        return img_in, img_out, cnt

    def compare_accuracy(v_xs, v_ys):
        # global predict
        y_pre = sess.run(predict, feed_dict={xs: v_xs, keep_prob: 1})
        pre = y_pre > 0.5
        pre = sess.run(tf.cast(pre, tf.float32))
        correct_predict = tf.equal(pre, v_ys)
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        result = sess.run(accuracy)
        return result

    def weight_variable(name, shape):
        return (tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()))

    def bias_variable(name, shape):
        return (tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()))

    def con2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pooling_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # define placeholder
    xs = tf.placeholder(tf.float32, [None, 2000])
    ys = tf.placeholder(tf.float32, [None, 2000])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    model_path = 'model.ckpt'

    # 5X5 patch, size:1 height:32
    W_conv1 = weight_variable('W1', [5, 5, 1, 16])
    b_conv1 = bias_variable('b1', [16])
    h_conv1 = tf.nn.relu(tf.nn.bias_add(con2d(x_image, W_conv1), b_conv1))  # output 100 * 20 * 16
    h_pool1 = max_pooling_2x2(h_conv1)  # output 50 * 10 * 16
    h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

    # conv layer2
    # 5X5 patch, size:1 height:32
    W_conv2 = weight_variable('W2', [5, 5, 16, 32])
    b_conv2 = bias_variable('b2', [32])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool1, W_conv2), b_conv2))  # output 50 * 10 * 32
    h_pool2 = max_pooling_2x2(h_conv2)  # output 25 * 5 * 32
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

    #deconv 1
    W_conv7 = weight_variable('W8', [5, 5, 16, 32])
    b_conv7 = bias_variable('b8', [16])
    deconv1 = tf.nn.conv2d_transpose(value=h_pool2, filter=W_conv7, output_shape=[batch_size, 10, 50, 16],
                                     strides=[1, 2, 2, 1], padding='SAME')
    deconv1 = tf.nn.relu(tf.nn.bias_add(deconv1, b_conv7))
    deconv1 = tf.nn.dropout(deconv1, keep_prob)

    W_conv8 = weight_variable('W9', [5, 5, 1, 16])
    b_conv8 = bias_variable('b9', [1])
    deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=W_conv8, output_shape=[batch_size, 20, 100, 1],
                                     strides=[1, 2, 2, 1], padding='SAME')
    deconv2 = tf.nn.bias_add(deconv2, b_conv8)
    predict = tf.reshape(deconv2, [-1, 100*20])

    cross = tf.reduce_mean(tf.pow(tf.subtract(predict, xs), 2.0))
    train = train_method(train_step).minimize(cross)

    saver = tf.train.Saver()

    img_in, img_out, cnt = read_and_decode('train.tfrecords')
    img_in_batch, img_out_batch, cnt_batch = tf.train.shuffle_batch([img_in, img_out, cnt], batch_size=batch_size, capacity=500,
                                                               min_after_dequeue=80, num_threads=1)
    if method == 'train':
        with tf.Session(graph = tf.get_default_graph()) as sess:
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            fp = open(LOG_PATH + current_time + ".txt", 'w+')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess=sess)
            pre_dict = 0
            for i in range(10000):
                img_in_val, img_out_val, cnt_val = sess.run([img_in_batch, img_out_batch, cnt_batch])
                np.set_printoptions(threshold=np.inf)
                sess.run(train, feed_dict={xs: img_in_val, ys: img_out_val, keep_prob: 0.5})
                if i % 5 == 0:
                    print('cnt: %d' % i)
                    str = 'cnt: %d' % i + '\n'
                    # print(label_t_val)
                    cross_sess = sess.run(cross, feed_dict={xs: img_in_val, ys: img_out_val, keep_prob: 1})
                    accuracy_sess = compare_accuracy(img_in_val, img_out_val)
                    print("cross_sess: %f" % cross_sess)
                    print("train accuracy: %f" % accuracy_sess)
                    str += "cross_sess: %f" %cross_sess + '\n'
                    str += "train accuracy: %f" %accuracy_sess + '\n'
                    if accuracy_sess > 0.99:
                        break
                    if i % 500 == 0:
                        print("pre cross: %f and current cross: %f" % (pre_dict, cross_sess))
                        str += "pre cross: %f and current cross: %f" %(pre_dict, cross_sess) + '\n'
                        if i != 0:
                            if pre_dict >= cross_sess + 0.002:
                                pre_dict = cross_sess
                            else:
                                pre_dict = cross_sess
                        else:
                            pre_dict = cross_sess
                    result_process(i, cross_sess, accuracy_sess)
                    fp.write(str)
                    fp.flush()
            saver.save(sess, model_path)
            coord.request_stop()
            coord.join(threads)
            fp.close()
    elif method == 'test':
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess=sess)
            saver.restore(sess, model_path)
            #for i in range(1):
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    def print_result(cnt, cross, accuracy):
        pass

    tf_ocr_train(tf.train.AdamOptimizer, 1e-3, print_result, method=sys.argv[1])

