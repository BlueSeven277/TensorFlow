import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# x^2 regression

#generate 200 point in range(-0.5,0.5)
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]#20*1 matrix
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+ noise

#define placeholder
x = tf.placeholder(tf.float32,[None,1]) #n*1 matrix; 200 is passed so None will be 200
y = tf.placeholder(tf.float32,[None,1])

#define  neural network intermediate layers
Weitht_l1 = tf.Variable(tf.random_normal([1,10]))
biases_l1 = tf.Variable(tf.zeros([1,10])) #initiallied by 0
Wx_plus_b_l1 = tf.matmul(x,Weitht_l1) + biases_l1
L1 = tf.nn.tanh(Wx_plus_b_l1)

# define output layer
Weitht_l2 = tf.Variable(tf.random_normal([10,1]))
biases_l2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_l2 = tf.matmul(L1, Weitht_l2) + biases_l2
prediction = tf.nn.tanh(Wx_plus_b_l2)

# define loss function
loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #0.1 learning rate

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range (2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    # get prediction values
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data,prediction_value,'r', lw =5)
    plt.show()