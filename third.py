# hand writing numbers regonition

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples //batch_size

#define place holders
x = tf.placeholder(tf.float32,[None,784]) #every picture is 28*28=784 None = 100
y = tf.placeholder(tf.float32,[None,10])  #10 labels, one-hot

#define neural network
W = tf.Variable(tf.zeros([784,10])) #input layer 784 neurals , output 10 nrurals
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)


#define loss function
loss = tf.reduce_mean(tf.square(y-prediction))
#use gradient descent
train_step = tf.train.GradientDescentOptimizer(0.25).minimize(loss)
#initialization
init = tf.global_variables_initializer()

#evaluation
# get label position (10 positions in one-hot in this case) with the highest probability
# return True is the y label is same as the prediction label
# boolean list
correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
# change boolean list to float32, true =1, false =0, add together and calculate average as accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range (55):
        for batch in range(n_batch):
            batch_xs, batchys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batchys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print ("iterator" + str(epoch)+ ", Testing Accuracy "+ str(acc))