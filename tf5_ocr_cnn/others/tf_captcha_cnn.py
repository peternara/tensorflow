import tensorflow as tf
import numpy as np
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

IMAGE_HEIGHT =  25#114
IMAGE_WIDTH = 100#450
MAX_CAPTCHA = 6
CHAR_SET_LEN = 26
BATCH_SIZE = 3429

def tf_ocr_train(train_method, train_step, result_process, method='train'):
    global predict
    def read_and_decode(tf_record_path):  # read iris_contact.tfrecords
        filename_queue = tf.train.string_input_producer([tf_record_path])  # create a queue

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # return file_name and file
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'cnt': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.string),
                                           })  # return image and label

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [IMAGE_HEIGHT*IMAGE_WIDTH])  # reshape image
        img = tf.cast(img, tf.float32) * (1. / 255)
        label = tf.decode_raw(features['label'], tf.uint8)
        label = tf.reshape(label, [MAX_CAPTCHA*CHAR_SET_LEN])
        label = tf.cast(label, tf.float32)
        cnt = features['cnt']  # throw label tensor
        return img, label, cnt

    def compare_accuracy(v_xs, v_ys):
        #global predict
        y_pre = sess.run(predict, feed_dict={xs: v_xs, keep_prob: 1})
        y_pre = tf.reshape(y_pre, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
        max_idx_p = tf.argmax(y_pre, 2)
        max_idx_l = tf.argmax(tf.reshape(v_ys, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #print(sess.run(max_idx_l))
        #print(sess.run(max_idx_p))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
        return result

    def weight_variable(name, shape):
        # create random variable
        # truncated_normal (shape, mean, stddev)  gauss function
        #initial = 0.01 * tf.random_normal(shape)
        #initial = tf.truncated_normal(shape, stddev=0.1)
        #return tf.Variable(initial)
        return(tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()))

    def bias_variable(name, shape):
        #initial = tf.constant(0.1, shape=shape)
        #return tf.Variable(initial)
        return (tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()))

    def con2d(x, W):
        # strides[1,x_mov, y_mov,1]  step: x_mov = 1 y_mov = 1, stride[0]&stride[3] =1
        # input size is same with output same
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pooling_2x2(x):
        # step change to 2
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #define placeholder
    xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
    ys = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    model_path = 'model.ckpt'

    #conv layer1
    # 5X5 patch, size:1 height:32
    W_conv1 = weight_variable('W1', [5, 5, 1, 32])
    b_conv1 = bias_variable('b1', [32])
    h_conv1 = tf.nn.relu(tf.nn.bias_add(con2d(x_image, W_conv1), b_conv1))  # output 100 * 25 * 32
    h_pool1 = max_pooling_2x2(h_conv1)  # output 50 * 13 * 32
    h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

    #conv layer2
    # 5X5 patch, size:1 height:32
    W_conv2 = weight_variable('W2', [5, 5, 32, 64])
    b_conv2 = bias_variable('b2', [64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool1, W_conv2), b_conv2))  # output 50 * 13 * 64
    h_pool2 = max_pooling_2x2(h_conv2)  # output 25 * 7 * 64
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

    h_pool3_flat = tf.reshape(h_pool2, [-1, 25 * 7 * 64])

    '''W_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool2, W_conv3), b_conv3))  # output 113 * 29 * 64
    h_pool3 = max_pooling_2x2(h_conv3)  # output 57 * 15 * 64
    h_pool3 = tf.nn.dropout(h_pool3, keep_prob)
    #h_pool3_flat = tf.reshape(h_pool3, [-1, 57 * 15 * 64])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 13 * 4 * 64])'''

      # full connect layer4
    #W_fc4 = weight_variable([15 * 57 * 64, 1024])
    W_fc4 = weight_variable('W3', [25 * 7 * 64, 1024])
    b_fc4 = bias_variable('b3', [1024])
    h_fc4 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, W_fc4), b_fc4))
    h_fc4 = tf.nn.dropout(h_fc4, keep_prob)

    '''
    W_fc51 = weight_variable([1024, CHAR_SET_LEN])
    b_fc51 = bias_variable([CHAR_SET_LEN])
    h_fc51 = tf.nn.softmax(tf.add(tf.matmul(h_fc4, W_fc51), b_fc51))

    W_fc52 = weight_variable([1024, CHAR_SET_LEN])
    b_fc52 = bias_variable([CHAR_SET_LEN])
    h_fc52 = tf.nn.softmax(tf.add(tf.matmul(h_fc4, W_fc52), b_fc52))

    W_fc53 = weight_variable([1024, CHAR_SET_LEN])
    b_fc53 = bias_variable([CHAR_SET_LEN])
    h_fc53 = tf.nn.softmax(tf.add(tf.matmul(h_fc4, W_fc53), b_fc53))

    W_fc54 = weight_variable([1024, CHAR_SET_LEN])
    b_fc54 = bias_variable([CHAR_SET_LEN])
    h_fc54 = tf.nn.softmax(tf.add(tf.matmul(h_fc4, W_fc54), b_fc54))

    W_fc55 = weight_variable([1024, CHAR_SET_LEN])
    b_fc55 = bias_variable([CHAR_SET_LEN])
    h_fc55 = tf.nn.softmax(tf.add(tf.matmul(h_fc4, W_fc55), b_fc55))

    W_fc56 = weight_variable([1024, CHAR_SET_LEN])
    b_fc56 = bias_variable([CHAR_SET_LEN])
    h_fc56 = tf.nn.softmax(tf.add(tf.matmul(h_fc4, W_fc56), b_fc56))
    

    #full connect layer6
    predict = tf.concat([h_fc51, h_fc52, h_fc53, h_fc54, h_fc55, h_fc56], 1)'''

    #full connect layer5
    W_fc5 = weight_variable('W4', [1024, MAX_CAPTCHA * CHAR_SET_LEN])
    b_fc5 = bias_variable('b4', [MAX_CAPTCHA * CHAR_SET_LEN])
    predict = tf.add(tf.matmul(h_fc4, W_fc5), b_fc5)

    cross = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=ys))
    train = train_method(train_step).minimize(cross)

    saver = tf.train.Saver()

    img, label, cnt = read_and_decode('train.tfrecords')
    #img_batch, label_batch, cnt_batch = tf.train.batch([img, label, cnt], batch_size=BATCH_SIZE,
                               #capacity=BATCH_SIZE)
    img_batch, label_batch, cnt_batch = tf.train.shuffle_batch([img, label, cnt], batch_size=200,
                                                               capacity=3000, min_after_dequeue=1000, num_threads=1)

    img_t, label_t, cnt_t = read_and_decode('test.tfrecords')
    img_t_batch, label_t_batch, cnt_t_batch = tf.train.batch([img_t, label_t, cnt_t], batch_size=1,
                                capacity=10)
    if method == 'train':
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess=sess)
            pre_dict = 0
            cross_sess = 0
            for i in range(10000):
                img_val, label_val, cnt_val = sess.run([img_batch, label_batch, cnt_batch])
                np.set_printoptions(threshold=np.inf)
                sess.run(train, feed_dict={xs: img_val, ys: label_val, keep_prob:0.5})
                if i%1 == 0:
                    img_t_val, label_t_val, cnt_t_val = sess.run([img_t_batch, label_t_batch, cnt_t_batch])
                    cross_sess = sess.run(cross, feed_dict={xs: img_val, ys: label_val, keep_prob: 1})
                    accuracy_sess = compare_accuracy(img_val, label_val)
                    print('--------------------')
                    print("%d:" %i)
                    print("cross: %f" %cross_sess)
                    print("train accuracy: %f" %accuracy_sess)
                    accuracy_sess = compare_accuracy(img_t_val, label_t_val)
                    result_process(i, cross_sess, accuracy_sess)
                    print("test accuracy: %f" % accuracy_sess)
                    if accuracy_sess > 0.99:
                        break
                if i%500 == 0:
                    print(cross_sess)
                    print("pre value: %f and current value: %f" %(pre_dict, cross_sess))
                    if i != 0:
                        if pre_dict >= cross_sess + 0.002:
                            pre_dict = cross_sess
                        else:
                            break
                    else:
                        pre_dict = cross_sess
            saver.save(sess, model_path)
            coord.request_stop()
            coord.join(threads)
    elif method == 'test':
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess=sess)
            saver.restore(sess, model_path)
            for i in range(10):
                img_t_val, label_t_val, cnt_t_val = sess.run([img_t_batch, label_t_batch, cnt_t_batch])
                accuracy_sess = compare_accuracy(img_t_val, label_t_val)
                print(accuracy_sess)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    def print_result(cnt, cross, accuracy):
        pass
    #    print('%d:' %cnt)
    #    print(cross)
    #    print(accuracy)
    tf_ocr_train(tf.train.AdamOptimizer, 1e-3, print_result, method=sys.argv[1])
