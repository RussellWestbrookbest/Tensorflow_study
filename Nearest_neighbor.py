# Author: ye

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

"""import the data set"""
mnist = input_data.read_data_sets('/Users/yangyimin/PycharmProjects/Tensorflow', one_hot=True)
Xtrain, Ytrain = mnist.train.next_batch(5000)
Xtest, Ytest = mnist.test.next_batch(200)

"""initiate the parameters"""
xtrain, xtest = tf.placeholder('float32', [None, 784]), tf.placeholder('float32', [784])

"""calculate the distance"""
distance = tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))), reduction_indices=1)
prediction = tf.arg_min(distance, 0)
accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Xtest)):
        index = sess.run(prediction, feed_dict={xtrain: Xtrain, xtest: Xtest[i, :]})
        print('Test', i, 'prediction: ', np.argmax(Ytrain[index]), 'Result: ', np.argmax(Ytest[i]))
        if np.argmax(Ytrain[index]) == np.argmax(Ytest[i]):
            accuracy += 1 / len(Xtest)

    print('Accuracy is {}'.format(accuracy))




