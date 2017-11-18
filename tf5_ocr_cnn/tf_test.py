import tensorflow as tf
import numpy as np

x = tf.Variable(2)
y = tf.Variable(3)
z = tf.add(x, y)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(0, 10):
        t = sess.run(z)
        print(t)
    print(sess.graph.as_graph_def())
