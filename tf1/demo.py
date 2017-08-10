import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def add_layer(inputs, input_size, out_size, n_layer, activation_function=None):
    layer_name = 'layername%s' % n_layer
    #tensorboard display
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([input_size, out_size]), name='W')
            #tensorboard train change graph display
            tf.summary.histogram(layer_name + 'weights', Weights)
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + 'bias', bias)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + bias
        #activation function: output =f(W*x+b)
        if activation_function == None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + 'output', outputs)
        return  outputs

#linespace creat n=300, [-1, 1] num list
#np.newaxis: eg. x = [1,2,3]  so x[:,np.newaxis] = [1,2,3]T
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#random.normal, gaussian noise,  mean = 0, variance = 0.05
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope("inputs"):
    #placeholder: variable with not assigned value
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
o2 = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope("loss"):
    #reduce_summ choose which dimension to sum  which given in reduction_indices
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - o2),
                     reduction_indices=[1]),
                     name = 'loss')
    #tensorboard graph scalar to display loss
    tf.summary.scalar('loss', loss)

with tf.name_scope("train"):
    #use GradientDescent to minimize loss
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#init all variables, it will be activated by use sess.run(init)
init = tf.global_variables_initializer()

sess = tf.Session()
#merge all summmary(weight, bias, loss) to one variable
merged = tf.summary.merge_all()
#save tensorboard file to logs dir
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

#matlib plot function to display data input with output
fig = plt.figure()
#subplot(x, y, postion)
ax = fig.add_subplot(1,1,1)
#graph: xpos : xdata, ypos: ydata
ax.scatter(x_data, y_data)
#use ion to show onetime and disapear
plt.ion()
#show graphic
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        prediction_value = sess.run(o2, feed_dict={xs:x_data})
        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        #add summary graphc display(weight, bias, loss ..) to tensorboard files
        writer.add_summary(result, i)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        #draw line (x:xdata, y:ydata, color:red, step:5 point)
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        #delay 100 ms
        plt.pause(0.1)
#use ioff to show always time, not disapear
plt.ioff()
plt.show()

# how to use tensorboard ?
#1. tensorflow# tensorboard --log_dir='logs/'
#2. copy http://cloud:6006 to chrome and open
