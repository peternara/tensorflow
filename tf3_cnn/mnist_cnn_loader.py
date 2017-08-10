import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compare_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob:1})
    correct_predict = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result

def weight_variable(shape):
    #create random variable
    #truncated_normal (shape, mean, stddev)  gauss function
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def con2d(x, W):
    #strides[1,x_mov, y_mov,1]  step: x_mov = 1 y_mov = 1, stride[0]&stride[3] =1
    #input size is same with output same
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling_2x2(x):
    #step change to 2
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, 784]) # 28*28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

#5X5 patch, size:1 height:32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#conv1 layer
h_conv1 = tf.nn.relu(con2d(x_image, W_conv1) + b_conv1) # output 28 * 28 * 32
h_pool1 = max_pooling_2x2(h_conv1)  # output 14 * 14 * 32

#conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_con2 = tf.nn.relu(con2d(h_pool1, W_conv2) + b_conv2) #output 14*14*64
h_pool2 = max_pooling_2x2(h_con2)  #output 7 * 7 * 64

#func1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # [n_samples, 7, 7, 64] ==> [n_samples, 7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#func2_layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

saver = tf.train.Saver()
sess = tf.Session()

saver.restore(sess, "saver/save_cnn.ckpt")

print(compare_accuracy(
    mnist.test.images, mnist.test.labels))