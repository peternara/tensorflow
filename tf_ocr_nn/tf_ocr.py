import tensorflow as tf
import numpy as np

def tf_ocr_train(train_method, train_step, result_process):
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

    def add_layer(inputs, input_size, out_size, n_layer, activation_function=None):
        layer_name = 'layername%s' % n_layer
        with tf.name_scope('layer'):
            with tf.name_scope('Weight'):
                # Weights = tf.truncated_normal([input_size, out_size], stddev=0.1, name='W')
                Weights = tf.Variable(tf.random_normal([input_size, out_size]), name='W')
            with tf.name_scope('bias'):
                bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs, Weights) + bias
                # Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
            # activation function: output =f(W*x+b)
            if activation_function == None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            return outputs

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

    model_path = 'model.ckpt'

    #layer1
    W1 =   tf.Variable(tf.random_normal([400, 128]))
    bias1 = tf.Variable(tf.zeros([1, 128]) + 0.1)
    layer1 = tf.nn.sigmoid(tf.matmul(xs, W1) + bias1)

    #layer2
    W2 =  tf.Variable(tf.random_normal([128, 128]))
    bias2 = tf.Variable(tf.zeros([1, 128]) + 0.1)
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + bias2)

    #layer3
    W3 = tf.Variable(tf.random_normal([128, 54]))
    bias3 = tf.Variable(tf.zeros([1, 54]) + 0.1)
    predict = tf.nn.softmax(tf.matmul(layer2, W3) + bias3)

    #cross entropy
    cross = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(predict, 1e-10,1.0)), \
                reduction_indices=[1]))
    #cross = tf.reduce_mean(tf.reduce_mean(-tf.reduce_sum(ys*tf.log(predict), reduction_indices=[1])))
    #train = tf.train.AdamOptimizer(1e-4).minimize(cross)
    #train = tf.train.GradientDescentOptimizer(0.2).minimize(cross)

    train = train_method(train_step).minimize(cross)

    saver = tf.train.Saver()

    img, label, cnt = read_and_decode('train.tfrecords')
    #img_batch, label_batch, cnt_batch = tf.train.shuffle_batch([img, label, cnt], batch_size=1500,
    #                                capacity=2000, min_after_dequeue=1000, num_threads=2)
    #use full batch size
    img_batch, label_batch, cnt_batch = tf.train.batch([img, label, cnt], batch_size=2382,
                                capacity=2382)

    img_t, label_t, cnt_t = read_and_decode('test.tfrecords')
    img_t_batch, label_t_batch, cnt_t_batch = tf.train.batch([img_t, label_t, cnt_t], batch_size=270,
                                capacity=270)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)

        for i in range(10000):
            img_val, label_val, cnt_val = sess.run([img_batch, label_batch, cnt_batch])
            np.set_printoptions(threshold=np.inf)
            sess.run(train, feed_dict={xs: img_val, ys: label_val, keep_prob:1})
            if i%50 == 0:
                #print('%d:' %i)
                img_t_val, label_t_val, cnt_t_val = sess.run([img_t_batch, label_t_batch, cnt_t_batch])
                #print(label_t_val)
                cross_sess = sess.run(cross, feed_dict={xs: img_val, ys: label_val, keep_prob: 1})
                accuracy_sess = compare_accuracy(img_t_val, label_t_val)
                #print(sess.run(predict[0], feed_dict={xs: img_val, ys: label_val, keep_prob: 1}))
                #print(cross_sess)
                #print(accuracy_sess)
                #print(sess.run(bias1))
                #print(sess.run(bias2))
                result_process(i, cross_sess, accuracy_sess)
        saver.save(sess, model_path)
        sess.close()

if __name__ == '__main__':
    def print_result(cnt, cross, accuracy):
        print('%d:' %cnt)
        print(cross)
        print(accuracy)
    tf_ocr_train(tf.train.AdagradOptimizer, 0.2, print_result)
