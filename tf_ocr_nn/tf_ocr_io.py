import tensorflow as tf
import numpy as np
import tf_ocr
import matplotlib.pyplot as plt
import csv
import re

cross_title = ['times', 'gradient-0.05_cross', 'gradient-0.1_cross', 'gradient-0.2_cross', 'gradient-0.5_cross',
            'gradient-0.8_cross', 'adam-le-4_cross', 'adag-0.2_cross']

accuracy_title = ['', 'times', 'gradient-0.05_accuracy', 'gradient-0.1_accuracy', 'gradient-0.2_accuracy',
                  'gradient-0.5_accuracy', 'gradient-0.8_accuracy', 'adam-le-4_accuracy', 'adag-0.2_accuracy']

total_title = cross_title + accuracy_title

train_method_list = [tf.train.GradientDescentOptimizer,
                     tf.train.GradientDescentOptimizer,
                     tf.train.GradientDescentOptimizer,
                     tf.train.GradientDescentOptimizer,
                     tf.train.GradientDescentOptimizer,
                     tf.train.AdamOptimizer,
                     tf.train.AdagradOptimizer]

train_step_list = [0.05, 0.1, 0.2, 0.5, 0.8, 1e-4, 0.2]

csv_name = './result/result.csv'
result_name = './result/result.txt'
result_fig = './result/result.jpg'

def init():
    with open(csv_name, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=total_title)
        writer.writeheader()

    with open(result_name, 'w+') as f:
        for i in range(200):
            buf = 'times_' + str(i*50) + ',\n'
            f.writelines(buf)
        f.close()

def save_result(cnt, cross, accuracy):
    global line
    global buf
    print("%d: " %cnt)
    print(cross)
    print(accuracy)

    with open(result_name, 'r') as f:
        for line in f.readlines():
            if line.find('times_' + str(cnt)) >= 0:
                buf = line.rstrip() + str(cross) + ',' + str(accuracy) + ',\n'
                break
    open(result_name, 'r+').write(re.sub(line, buf, open(result_name).read()))

def save_csv(result_path, result_csv):
    global line
    global cross_list
    global accuracy_list
    global total_list
    with open(result_path, 'r') as f:
        with open(csv_name, 'a+') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for line in f.readlines():
                accuracy_list = []
                cross_list = []
                total_list = line.split(',')
                cross_list.append(total_list[0])
                for i, value in enumerate(total_list):
                    if i%2 == 0:
                        accuracy_list.append(value)
                    else:
                        cross_list.append(value)
                cross_list[-1] = ''
                print(accuracy_list)
                print(cross_list)
                csv_writer.writerow(cross_list + accuracy_list)

def plot_result(result_csv, result_plot):
    # matlib plot function to display data input with output
    fig = plt.figure()
    # subplot(x, y, postion)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title('training cross')
    plt.xlabel('times')
    plt.ylabel('cross')

    global x
    global y_list
    global line_list

    line_list = ['r-', 'r:', 'b-', 'b:', 'g-', 'g:', 'y-']
    y_list = np.zeros((7, 200))
    x = np.linspace(0, 10000, 200)
    for i in range(7):
        with open(result_csv,'r') as csvfile:
            reader = csv.DictReader(csvfile)
            column = [row[cross_title[i+1]] for row in reader]
            y_list[i] = np.array(column)
        ax.plot(x, y_list[i], line_list[i], lw=1, label=cross_title[i+1])

    #loc = location
    plt.legend(loc=1, fontsize="small")
    #plt.ioff()
    #show graphic
    #plt.show()
    plt.savefig(result_plot)




if __name__ == '__main__':
    init()
    for i in range(7):
        tf_ocr.tf_ocr_train(train_method_list[i], train_step_list[i], save_result, method='train')
    save_csv(result_name, csv_name)
    plot_result(csv_name, result_fig)