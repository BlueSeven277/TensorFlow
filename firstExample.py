import tensorflow as tf


#two op
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
#m1 mul m2 matrix product
product = tf.matmul(m1,m2)
print (product)
# initial a session
sess = tf.Session()
result = sess.run(product)
print (result)
sess.close()
with tf.Session() as sess:
    result = sess.run(product)
    print (result)

# the use of variables
x= tf.Variable([1,2])
a = tf.constant([3,3])
sub = tf.subtract(x,a)
add = tf.add(x,sub)
init = tf.global_variables_initializer()
#tf says error if variable not initialized
with tf.Session() as sess:
    sess.run(init)
    print (sess.run(sub))
    print (sess.run(add))


#loops
#variable is initialized by 0
state = tf.Variable(0,name = 'counter')
new_value = tf.add(state,1)
update = tf.assign(state,new_value)#assign function, use the value of new_value to assign state
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (sess.run(state))
    for _ in range(5):
        sess.run(update)
        print (sess.run(state))

# fetch and feed
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
add = tf.add(input2,input3)
mul = tf.multiply(input1,add)
with tf.Session() as sess:
    # fetch: execute multiple op tofether
    result = sess.run([mul,add])
    print (result)

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
out = tf.multiply(input1,input2)
with tf.Session() as sess:
    #feed: use placeholder first and pass value later
    print (sess.run(out, feed_dict={input1:[7.0],input2:[2.0]}))

# an easy instance example
import numpy as np
x_data = np.random.rand(100)
y_data = x_data*0.1+0.2

# build a linear model
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# build a square loss function, calculate loss
loss = tf.reduce_mean(tf.square(y_data-y))
# define a gradient descent to train optimizer
optimizer = tf.train.GradientDescentOptimizer(0.2) #learning rate = 0.2
# define a minimize loss function
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 ==0:
            print (step,sess.run([k,b]))