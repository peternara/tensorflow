import tensorflow as tf
import numpy as np
import tf_ocr
import matplotlib.pyplot as plt
import csv
import re

total_title = ['times', 'cross', 'test-accuracy', 'train-accuracy']

active_func = {
    'Grad-0.05': (
        tf.train.GradientDescentOptimizer, 0.05
    ),
    'Grad-0.1': (
        tf.train.GradientDescentOptimizer, 0.1
    ),
    'Grad-0.2': (
        tf.train.GradientDescentOptimizer, 0.2
    ),
    'Grad-0.5': (
        tf.train.GradientDescentOptimizer, 0.5
    ),
    'Grad-0.8': (
        tf.train.GradientDescentOptimizer, 0.8
    ),
    'Adam-1e-4': (
        tf.train.AdamOptimizer, 1e-4
    ),
    'Adag-0.2': (
        tf.train.AdagradOptimizer, 0.2
    ),
}


csv_name = './result/result.csv'
cross_fig = './result/cross.jpg'
train_accuracy_fig = './result/train_accuracy.jpg'
test_accuracy_fig = './result/test_accuracy.jpg'

def init():
    pass

def save_result(cnt, cross, accuracy, train_accuracy):
    print("%d: " %cnt)
    print('cross: %f' %cross)
    print('test accuracy: %f' %accuracy)
    print('train accuracy: %f' %train_accuracy)
    result_list = [cnt, cross, accuracy, train_accuracy]
    csv_writer.writerow(result_list)

def plot_result(result_plot):
    fig = plt.figure()
    # subplot(x, y, postion)
    ax = fig.add_subplot(1, 1, 1)

    line_list = ['r-', 'r:', 'b-', 'b:', 'g-', 'g:', 'y-']
    x = np.linspace(0, 3000, 60)
    i = 0

    if result_plot == './result/cross.jpg':
        ax.set_title('training cross')
        plt.xlabel('times')
        plt.ylabel('cross')

        for key, value in active_func.items():
            csv_name = './result/' + key + '.csv'
            with open(csv_name, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                column = [row[total_title[1]] for row in reader]
                ax.plot(x, np.array(column), line_list[i], lw=1, label=key)
            i += 1
        plt.legend(loc=1, fontsize="small")
        plt.savefig(result_plot)
    elif result_plot == './result/train_accuracy.jpg':
        ax.set_title('train accuracy')
        plt.xlabel('times')
        plt.ylabel('train accuracy')

        for key, value in active_func.items():
            csv_name = './result/' + key + '.csv'
            print(csv_name)
            with open(csv_name, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                column = [row[total_title[3]] for row in reader]
                ax.plot(x, np.array(column), line_list[i], lw=1, label=key)
            i += 1
        plt.legend(loc=1, fontsize="small")
        plt.savefig(result_plot)
    else:
        ax.set_title('test accuracy')
        plt.xlabel('times')
        plt.ylabel('test accuracy')

        for key, value in active_func.items():
            csv_name = './result/' + key + '.csv'
            print(csv_name)
            with open(csv_name, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                column = [row[total_title[2]] for row in reader]
                ax.plot(x, np.array(column), line_list[i], lw=1, label=key)
            i += 1
        plt.legend(loc=1, fontsize="small")
        plt.savefig(result_plot)


if __name__ == '__main__':
    init()
    i=0

    for key, value in active_func.items():
        print('Start ' + key + ' training...')
        csv_name = './result/' + key + '.csv'
        with open(csv_name, 'w') as csv_file:
            global csv_writer
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(total_title)
            tf_ocr.tf_ocr_train(value[0], value[1], save_result, method='train', cnt=i)
        i += 1

    plot_result(cross_fig)
    plot_result(train_accuracy_fig)
    plot_result(test_accuracy_fig)
