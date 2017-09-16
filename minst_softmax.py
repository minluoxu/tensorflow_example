<<<<<<< HEAD
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
=======
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
minist = input_data.read_data_sets("MINIST_data/",one_hot=True)

print minist.train.images.shape, minist.train.labels.shape

print minist.test.images.shape, minist.test.labels.shape

print minist.validation.images.shape, minist.validation.labels.shape

import tensorflow as tf 
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784])
W  = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step  = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(30000):    
    if i%1000 == 0:
        print i
    batch_xs, batch_ys = minist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: minist.test.images,
                                      y_: minist.test.labels}))
>>>>>>> b69f7c41a08d268586cc37b6a331062e54fb4eab
