import numpy as np 
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math
import os
import cv2

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def plotNNFilter(units, title='Filter'):
    # filters = units.shape[3]
    # plt.figure(1, figsize=(39,30))
    # n_columns = 6
    # n_rows = math.ceil(filters / n_columns) + 1
    # for i in range(filters):
    #     plt.subplot(n_rows, n_columns, i+1)
    #     plt.title(title + '_' + str(i))
    #     image = units[0,:,:,i]
    #     plt.imshow(image, interpolation="nearest", cmap="gray")
    #     image_np = np.array(image)
    #     print(image_np)
    #     h,w = image_np.shape
    #     cArray1 = cv2.CreateMat(h, w, cv2.CV_32FC3)
    #     cArray2 = cv2.fromarray(image_np)
    #     cv2.CvtColor(cArray2, cArray1, cv2.CV_GRAY2BGR)
    #     cv2.imwrite(title + '_' + str(i) + '.png', cArray1)
    # plt.savefig(title + '.png')
    filters = units.shape[3]
    spines = 'left', 'right', 'top', 'bottom'
    labels = ['label' + spine for spine in spines]

    tick_params = {spine : False for spine in spines}
    tick_params.update({label : False for label in labels})

    for i in range(filters):
        print(i)
        image = units[0,:,:,i]
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        img = ax.imshow(image, cmap='magma', interpolation='nearest')
        for spine in spines:
            ax.spines[spine].set_visible(False)
        ax.tick_params(**tick_params)

        fig.savefig('test/' + title + '_' + str(i) + '.png', bbox_inches='tight', pad_inches=0, transparent=True)


def getActivatedUnits(layer, stimuli):
    return sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})


def plotActivatedUnits(layer,stimuli,title):
    plt.title(title)
    plotNNFilter(getActivatedUnits(layer, stimuli),title)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print (mnist)
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784],name="x-in")
true_y = tf.placeholder(tf.float32, [None, 10],name="y-in")
keep_prob = tf.placeholder("float")

x_image = tf.reshape(x,[-1,28,28,1])
hidden_1 = slim.conv2d(x_image,6,[5,5])
pool_1 = slim.max_pool2d(hidden_1,[2,2])
hidden_2 = slim.conv2d(pool_1,16,[5,5])
pool_2 = slim.max_pool2d(hidden_2,[2,2])
hidden_3 = slim.conv2d(pool_2,20,[5,5])
# hidden_3 = slim.dropout(hidden_3,keep_prob)

flatten = slim.flatten(hidden_3)
fc_1 = slim.fully_connected(flatten,40,activation_fn=tf.nn.relu)
# fc_2 = slim.fully_connected(fc_1,30,activation_fn=tf.nn.relu)
out_y = slim.fully_connected(fc_1,10,activation_fn=tf.nn.softmax)

cross_entropy = -tf.reduce_sum(true_y*tf.log(out_y))
correct_prediction = tf.equal(tf.argmax(out_y,1), tf.argmax(true_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

batchSize = 100

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
for i in range(701):
    batch = mnist.train.next_batch(batchSize)
    sess.run(train_step, feed_dict={x:batch[0],true_y:batch[1], keep_prob:0.5})
    if i % 100 == 0 and i != 0:
        trainAccuracy = sess.run(accuracy, feed_dict={x:batch[0],true_y:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, trainAccuracy))

testAccuracy = sess.run(accuracy, feed_dict={x:mnist.test.images,true_y:mnist.test.labels, keep_prob:1.0})
print("test accuracy %g"%(testAccuracy))

while True:
    try:
        id = int(input('select image:'))
        imageToUse = mnist.test.images[id]
        plt.imshow(np.reshape(imageToUse,[28,28]), interpolation="nearest", cmap="gray")
        plt.show()

        plotActivatedUnits(hidden_1, imageToUse, 'ConvL1')
        # plt.show()

        plotActivatedUnits(pool_1, imageToUse, 'PoolL1')
        # plt.show()

        # plotActivatedUnits(hidden_2, imageToUse, 'ConvL2')
        # # plt.show()

        # plotActivatedUnits(pool_2, imageToUse, 'PoolL2')
        # # plt.show()

        # plotActivatedUnits(hidden_3, imageToUse, 'ConvL3')
        # plt.show()

        # print(getActivatedUnits(fc_1, imageToUse))
        # print(getActivatedUnits(fc_2, imageToUse))
        print(getActivatedUnits(out_y, imageToUse))
    except Exception as e:
        print(e)
        pass
