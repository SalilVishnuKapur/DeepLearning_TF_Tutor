import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#################################################
# The main two purposes of Tensorflow are :-    #
# 1. To create the tensorflow graph             #   
# 2. To run the computation of tensorflow graph #
#################################################


# Read the data from MNIST
# This reading process does not involve tensorflow
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


# (Convolutional_layer + pooling) => 1
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], dtype = tf.float64))
b_conv1 = tf.Variable(tf.zeros([32], dtype= tf.float64))
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


# (Convolutional_layer + pooling) => 2
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32, 64], dtype = tf.float64))
b_conv2 = tf.variable(tf.Zeros([64], dtype = tf.float64))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides = [1,1,1,1], padding = 'SAME'), b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


# (Convolutional_layer + pooling) => 3
W_conv3 = tf.Variable(tf.truncated_normal([5,5,64, 128], dtype = tf.float64))
b_conv3 = tf.Variable(tf.zeros([128], dtype = tf.float64))
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides = [1,1,1,1], padding = 'SAME'), b_conv3)
h_pool3 = tf.nn.max_pool(h_conv3, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = 'SAME')


# (Fully Connected layer(1024 Neurons)) => 1
W_fc1 = weight_variable([14 * 14 * 32, 1024])
b_fc1 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool3, [-1, 14*14*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)


# (Fully Connected layer(10 Neurons)) => 2
w_out = tf.Variable(tf.truncated_normal([1024,10], dtype = tf.float64))
b_out = tf.Variable(tf.zeros([10], dtype = tf.float64))
h_out = tf.matmul(h_fc1, w_out) + b_out 

# Dropout layer
#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  

# Cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = h_out))


# Find cost and optimizing the weights
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Running the whole process through session
arrTrain = np.ndarray((22,1), dtype = float)
arrTest = np.ndarray((22,1), dtype = float)
count = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2200):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            arrTrain[count] = train_accuracy
            count+=1
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    count = 0 
    for i in range(2200):
        batch = mnist.test.next_batch(50)
        if i % 100 == 0:
            test_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('test accuracy %g' % test_accuracy)
            arrTest[count] = test_accuracy
            count+=1

