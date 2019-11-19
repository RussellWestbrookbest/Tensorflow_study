# Author: ye

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""import the data set"""
mnist = input_data.read_data_sets('/Users/yangyimin/PycharmProjects/Tensorflow', one_hot=True)

"""initiate the parameters"""
learning_rate = 0.01
epochs = 10
step = 1
batch_size = 100

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32', [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

cost = tf.reduce_mean(- tf.reduce_sum(y * tf.log(prediction), reduction_indices=1))
optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        average_cost = 0
        batches = int(mnist.train.num_examples / batch_size)
        for batch in range(batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss = sess.run([optimize, cost], feed_dict={x: batch_x, y: batch_y})
            average_cost += loss / batches
        if (epoch + 1) % step == 0:
            print('Epoch=', '%03d' % (epoch + 1), 'cost=', '{:.9f}'.format(average_cost))

    """test the tensorflow model"""
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))
    print('accuracy:{}'.format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))
