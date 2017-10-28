import tensorflow as tf
import numpy as np
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
        img = tf.reshape(img, [2000])  # reshape image
        img = tf.cast(img, tf.float32) * (1. / 255)
        label = tf.decode_raw(features['label'], tf.uint8)
        label = tf.reshape(label, [324])
        label = tf.cast(label, tf.float32)
        cnt = features['cnt']  # throw label tensor
        return img, label, cnt

    def compare_accuracy(v_xs, v_ys):
        #global predict
        y_pre = sess.run(predict, feed_dict={xs: v_xs, keep_prob: 1})
        y_pre = tf.reshape(y_pre, [-1, 54, 6])
        max_idx_p = tf.argmax(y_pre, 1)
        max_idx_l = tf.argmax(tf.reshape(v_ys, [-1, 54, 6]), 1)
        #print(sess.run(y_pre))
        #print(sess.run(max_idx_p))
        #print(sess.run(max_idx_l))
        correct_predict = tf.equal(max_idx_p, max_idx_l)
        #print(sess.run(correct_predict))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
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
    xs = tf.placeholder(tf.float32, [None, 2000])
    ys = tf.placeholder(tf.float32, [None, 324])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 100, 20, 1])

    model_path = 'model.ckpt'

    '''W_conv1 = weight_variable([1, 1, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(tf.nn.bias_add(con2d(x_image, W_conv1), b_conv1))
    h_pool1 = max_pooling_2x2(h_conv1)  # output 50 * 10 * 16
    h_pool2 = tf.nn.dropout(h_pool1, keep_prob)'''

    '''#conv layer1
    W_conv11 = weight_variable([1, 1, 1, 1])
    b_conv11 = bias_variable([1])
    h_conv11 = tf.nn.relu(tf.nn.bias_add(con2d(x_image, W_conv11), b_conv11))  # output 100 * 20 * 16
    h_pool11 = tf.nn.dropout(h_conv11, keep_prob)'''

    # 5X5 patch, size:1 height:32
    W_conv1 = weight_variable('W1', [5, 5, 1, 8])
    b_conv1 = bias_variable('b1', [8])
    h_conv1 = tf.nn.relu(tf.nn.bias_add(con2d(x_image, W_conv1), b_conv1))  # output 100 * 20 * 16
    h_pool1 = max_pooling_2x2(h_conv1)  # output 50 * 10 * 16
    h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

    #conv layer2
    # 5X5 patch, size:1 height:32
    W_conv2 = weight_variable('W2', [5, 5, 8, 16])
    b_conv2 = bias_variable('b2', [16])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool1, W_conv2), b_conv2))  # output 50 * 10 * 32
    h_pool2 = max_pooling_2x2(h_conv2)  # output 25 * 5 * 32
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 5 * 16])

    #conv layer2
    # 5X5 patch, size:1 height:32
    '''W_conv3 = weight_variable('W3', [5, 5, 64, 64])
    b_conv3 = bias_variable('b3', [64])
    h_conv3 = tf.nn.relu(tf.nn.bias_add(con2d(h_pool2, W_conv3), b_conv3))  # output 50 * 10 * 32
    h_conv3 = tf.nn.dropout(h_conv3, keep_prob)
    h_pool2_flat = tf.reshape(h_conv3, [-1, 25 * 5 * 64])
    '''

    # full connect layer3
    W_fc3 = weight_variable('W4', [25 * 5 * 16, 1024])
    b_fc3 = bias_variable('b4', [1024])
    h_fc3 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, W_fc3), b_fc3))
    h_fc3 = tf.nn.dropout(h_fc3, keep_prob)

    '''
    #full connect layer4
    W_fc4 = weight_variable('W5', [1024, 324])
    b_fc4 = bias_variable('b5', [324])
    predict = tf.add(tf.matmul(h_fc3, W_fc4), b_fc4)

    '''
    # full connect layer4-1
    W_fc4_1 = weight_variable('W51', [1024, 54])
    b_fc4_1 = bias_variable('b51', [54])
    fc4_1 = tf.nn.softmax(tf.matmul(h_fc3, W_fc4_1) + b_fc4_1)
    #fc4_1 = tf.matmul(h_fc3, W_fc4_1) + b_fc4_1
    # full connect layer4-2
    W_fc4_2 = weight_variable('W52', [1024, 54])
    b_fc4_2 = bias_variable('b52', [54])
    fc4_2 = tf.nn.softmax(tf.matmul(h_fc3, W_fc4_2) + b_fc4_2)
    #fc4_2 = tf.matmul(h_fc3, W_fc4_2) + b_fc4_2    
 
    # full connect layer4-3
    W_fc4_3 = weight_variable('W53', [1024, 54])
    b_fc4_3 = bias_variable('b53', [54])
    fc4_3 = tf.nn.softmax(tf.matmul(h_fc3, W_fc4_3) + b_fc4_3)
    #fc4_3 = tf.matmul(h_fc3, W_fc4_3) + b_fc4_3

    # full connect layer4-4
    W_fc4_4 = weight_variable('W54', [1024, 54])
    b_fc4_4 = bias_variable('b54', [54])
    fc4_4 = tf.nn.softmax(tf.matmul(h_fc3, W_fc4_4) + b_fc4_4)
    #fc4_4 = tf.matmul(h_fc3, W_fc4_4) + b_fc4_4

    # full connect layer4-5
    W_fc4_5 = weight_variable('W55', [1024, 54])
    b_fc4_5 = bias_variable('b55', [54])
    fc4_5 = tf.nn.softmax(tf.matmul(h_fc3, W_fc4_5) + b_fc4_5)
    #fc4_5 = tf.matmul(h_fc3, W_fc4_5) + b_fc4_5

    # full connect layer4-6
    W_fc4_6 = weight_variable('W56', [1024, 54])
    b_fc4_6 = bias_variable('b56', [54])
    fc4_6 = tf.nn.softmax(tf.matmul(h_fc3, W_fc4_6) + b_fc4_6)
    #fc4_6 = tf.matmul(h_fc3, W_fc4_6) + b_fc4_6
    
    predict = tf.concat([fc4_1, fc4_2, fc4_3, fc4_4, fc4_5, fc4_6], 1)

  
    #cross entropy
    #cross = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=ys))
    cross = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(predict, 1e-10,1.0)), \
               reduction_indices=[1]))
    #cross = tf.reduce_mean(tf.reduce_mean(-tf.reduce_sum(ys*tf.log(predict), reduction_indices=[1])))
    #train = tf.train.AdamOptimizer(1e-4).minimize(cross)
    #train = tf.train.GradientDescentOptimizer(0.2).minimize(cross)

    train = train_method(train_step).minimize(cross)

    saver = tf.train.Saver()

    img, label, cnt = read_and_decode('train.tfrecords')
    img_batch, label_batch, cnt_batch = tf.train.shuffle_batch([img, label, cnt], batch_size=200, capacity=6000, min_after_dequeue=4000, num_threads=1)
    #use full batch size
    #img_batch, label_batch, cnt_batch = tf.train.batch([img, label, cnt], batch_size=586, capacity=586)

    img_t, label_t, cnt_t = read_and_decode('test.tfrecords')
    img_t_batch, label_t_batch, cnt_t_batch = tf.train.shuffle_batch([img_t, label_t, cnt_t], batch_size=20, capacity=500, min_after_dequeue=400, num_threads=1)
    #img_t_batch, label_t_batch, cnt_t_batch = tf.train.batch([img_t, label_t, cnt_t], batch_size=935, capacity=935)
    if method == 'train':
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess=sess)
            pre_dict = 0
            for i in range(10000):
                img_val, label_val, cnt_val = sess.run([img_batch, label_batch, cnt_batch])
                np.set_printoptions(threshold=np.inf)
                sess.run(train, feed_dict={xs: img_val, ys: label_val, keep_prob:0.5})
                if i%5 == 0:
                    print('cnt: %d' %i)
                    img_t_val, label_t_val, cnt_t_val = sess.run([img_t_batch, label_t_batch, cnt_t_batch])
                    #print(label_t_val)
                    cross_sess = sess.run(cross, feed_dict={xs: img_val, ys: label_val, keep_prob: 1})
                    accuracy_sess = compare_accuracy(img_val, label_val)
                    print("cross_sess: %f" %cross_sess)
                    print("train accuracy: %f" %accuracy_sess)
                    accuracy_sess = compare_accuracy(img_t_val, label_t_val)
                    print("test accuracy: %f" %accuracy_sess)
                    if accuracy_sess > 0.99:
                        break
                    if i%500 == 0:
                        print("pre cross: %f and current cross: %f" %(pre_dict, cross_sess))
                        if i != 0:
                            if pre_dict >= cross_sess + 0.002:
                                pre_dict = cross_sess
                            else:
                                pre_dict = cross_sess
                        else:
                             pre_dict = cross_sess 
                    #print(sess.run(predict[0], feed_dict={xs: img_val, ys: label_val, keep_prob: 1}))
                    #print(cross_sess)
                    #print(accuracy_sess)
                    #print(sess.run(bias1))
                    #print(sess.run(bias2))
                    result_process(i, cross_sess, accuracy_sess)
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
        #print('%d:' %cnt)
        #print(cross)
        #print(accuracy)
    tf_ocr_train(tf.train.AdamOptimizer, 1e-3, print_result, method=sys.argv[1])
    #tf_ocr_train(tf.train.AdagradOptimizer, 0.2, print_result, method=sys.argv[1])
