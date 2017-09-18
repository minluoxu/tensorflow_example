# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as  tf 

# the network is to recognize the image use mnist dataset
# the struct network is two layer network have one hidden layer.
#  

# Import data
mnist_data = input_data.read_data_sets("MINIST_data/",one_hot=True)
print mnist_data.train.images.shape

# # Create the model
in_unit = 784
h1_unit = 300

W1 = tf.Variable(tf.truncated_normal([in_unit,h1_unit],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_unit]))
W2 = tf.Variable(tf.zeros([h1_unit,10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None,in_unit])
keep_prob = tf.placeholder(tf.float32)

# the hide layer

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
y_ = tf.placeholder(tf.float32, [None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# train the model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(4000):
    if i%1000 == 0:
        print i
    batch_xs, batch_ys = mnist_data.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob:0.75})
    

# Test trained model.
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist_data.test.images,
                                      y_: mnist_data.test.labels,keep_prob:1.0}))
