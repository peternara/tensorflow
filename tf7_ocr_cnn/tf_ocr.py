import tensorflow as tf
import numpy as np
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def tf_ocr_train(train_method, train_step, result_process, method='train', cnt=1):
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
        img = tf.reshape(img, [400])  # reshape image
        img = tf.cast(img, tf.float32) * (1. / 255)
        label = tf.decode_raw(features['label'], tf.uint8)
        label = tf.reshape(label, [54])
        label = tf.cast(label, tf.float32)
        cnt = features['cnt']  # throw label tensor
        return img, label, cnt

    def weight_variable(name, shape, reuse):
        # create random variable
        # truncated_normal (shape, mean, stddev)  gauss function
        #initial = 0.01 * tf.random_normal(shape)
        #initial = tf.truncated_normal(shape, stddev=0.1)
        #return tf.Variable(initial)
        with tf.variable_scope(name, reuse=reuse):
            return(tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()))

    def bias_variable(name, shape, reuse):
        #initial = tf.constant(0.1, shape=shape)
        #return tf.Variable(initial)
        with tf.variable_scope(name, reuse=reuse):
            return (tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()))

    def con2d(x, W):
        # strides[1,x_mov, y_mov,1]  step: x_mov = 1 y_mov = 1, stride[0]&stride[3] =1
        # input size is same with output same
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pooling_2x2(x):
        # step change to 2
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def compare_accuracy(v_xs, v_ys):
        #global predict
        y_pre = sess.run(predict, feed_dict={xs: v_xs, keep_prob: 1})
        correct_predict = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
        return result

    #define placeholder
    xs = tf.placeholder(tf.float32, [None, 400])
    ys = tf.placeholder(tf.float32, [None, 54])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 20, 20, 1])

    model_path = 'model.ckpt'

    # 5X5 patch, size:1 height:32
    W_conv1 = weight_variable('W1', [5, 5, 1, 16], (cnt != 0))
    b_conv1 = bias_variable('b1', [16], (cnt != 0))
    h_conv1 = tf.nn.relu(tf.nn.bias_add(con2d(x_image, W_conv1), b_conv1))  # output 20 * 20 * 16
    h_pool1 = max_pooling_2x2(h_conv1)  # output 10 * 10 * 16
    h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

    #conv layer2
    # 5X5 patch, size:1 height:32
    W_conv2 = weight_variable('W2', [5, 5, 16, 32], (cnt != 0))
    b_conv2 = bias_variable('b2', [32], (cnt != 0))
    h_conv2 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool1, W_conv2), b_conv2))  # output 10 * 10 * 32
    h_pool2 = max_pooling_2x2(h_conv2)  # output 5 * 5 * 32
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 32])

    # full connect layer3
    W_fc3 = weight_variable('W4', [5 * 5 * 32, 512], (cnt != 0))
    b_fc3 = bias_variable('b4', [512], (cnt != 0))
    h_fc3 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, W_fc3), b_fc3))
    h_fc3 = tf.nn.dropout(h_fc3, keep_prob)

    # full connect layer4
    W_fc4 = weight_variable('W5', [512, 54], (cnt != 0))
    b_fc4 = bias_variable('b5', [54], (cnt != 0))
    predict = tf.nn.softmax(tf.add(tf.matmul(h_fc3, W_fc4), b_fc4))

    #cross entropy
    cross = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(predict, 1e-10,1.0)), \
               reduction_indices=[1]))

    train = train_method(train_step).minimize(cross)

    saver = tf.train.Saver()

    img, label, cnt = read_and_decode('train.tfrecords')
    img_batch, label_batch, cnt_batch = tf.train.shuffle_batch([img, label, cnt], batch_size=200,
                                    capacity=2000, min_after_dequeue=1000, num_threads=1)
    #use full batch size
    #img_batch, label_batch, cnt_batch = tf.train.batch([img, label, cnt], batch_size=2382,
    #                            capacity=2382)

    img_t, label_t, cnt_t = read_and_decode('test.tfrecords')
    img_t_batch, label_t_batch, cnt_t_batch = tf.train.batch([img_t, label_t, cnt_t], batch_size=270,
                                capacity=270)
    if method == 'train':
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess=sess)

            for i in range(3000):
                img_val, label_val, cnt_val = sess.run([img_batch, label_batch, cnt_batch])
                np.set_printoptions(threshold=np.inf)
                sess.run(train, feed_dict={xs: img_val, ys: label_val, keep_prob:0.5})
                if i%50 == 0:
                    img_t_val, label_t_val, cnt_t_val = sess.run([img_t_batch, label_t_batch, cnt_t_batch])
                    cross_sess = sess.run(cross, feed_dict={xs: img_val, ys: label_val, keep_prob: 1})
                    accuracy_sess = compare_accuracy(img_t_val, label_t_val)
                    train_accuracy_sess = compare_accuracy(img_val, label_val)
                    result_process(i, cross_sess, accuracy_sess, train_accuracy_sess)
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
    def print_result(cnt, cross, accuracy, train_accuracy):
        print('%d:' %cnt)
        print(cross)
        print(accuracy)
        print(train_accuracy)
    tf_ocr_train(tf.train.AdamOptimizer, 1e-3, print_result, method=sys.argv[1])
