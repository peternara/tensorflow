import tensorflow as ts
import numpy as np

import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compare_accuracy(v_xs, v_ys):
    global predict
    y_pre = sess.run(predict, feed_dict={xs: v_xs, keep_prob:1})
    correct_predict = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result

def add_layer(inputs, input_size, out_size, n_layer, activation_function=None):
    layer_name = 'layername%s' % n_layer
    #tensorboard display
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([input_size, out_size]), name='W')
            #tensorboard train change graph display
            #tf.summary.histogram(layer_name + 'weights', Weights)
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            #tf.summary.histogram(layer_name + 'bias', bias)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + bias
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        #activation function: output =f(W*x+b)
        if activation_function == None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        #tf.summary.histogram(layer_name + 'output', outputs)
        return  outputs

#define placeholder
#input char image: 28*28 = 784, output :0-9
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
predict = add_layer(xs, 784, 10, n_layer=1, activation_function=tf.nn.softmax)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "saver/mnist_save.ckpt")

print(compare_accuracy(
    mnist.test.images, mnist.test.labels))