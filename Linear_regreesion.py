import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


""":parameter"""
learning_rate = 0.01
n = np.random
epochs = 100

data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])

X = data['Population'].tolist()
Y = data['Profit'].tolist()

W = tf.Variable(n.randn(), dtype='float32')
b = tf.Variable(n.randn(), dtype='float32')

prediction = tf.add(tf.multiply(X, W), b)

cost = tf.math.reduce_mean(tf.square(prediction - Y))
train = tf.train.GradientDescentOptimizer(learning_rate)
optimize = train.minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        sess.run(optimize)
        print('Epoch = %d, cost = %f, [w=%f, b=%f]' % (epoch, sess.run(cost), sess.run(W), sess.run(b)))
    plt.plot(X, Y, 'bo')
    plt.plot(X, [i * sess.run(W) + sess.run(b) for i in X])
    plt.show()
















