#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
'''
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for epoch in range(101):
    for (_x, _y) in zip(x_data, y_data):
        sess.run(train, feed_dict={X: _x, Y: _y})
    if epoch%5000==0:
        print("# cost: ", sess.run(cost, feed_dict={X: x_data, Y: y_data}))
#ValueError: Cannot feed value of shape (0, 31399, 50) for Tensor 'Placeholder_24:0', which has shape '(1, 50)'
'''


# In[2]:


import csv
from math import sin

a = []
A = 0.1
i = A
cnt = i
while i<=3.14:
    q = []
    q.append(cnt)
    for j in range(2,11,2):
        #ins = round(i**j,j)
        q.append(i**j)
    q.append(sin(i))
    
    a.append(q)
    #print(q)
    #print('##',a)
    i += A
    cnt += A

with open('sinL.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in a:
        writer.writerow(i)


# In[3]:


init = tf.compat.v1.global_variables_initializer()


# In[4]:


data = pd.read_csv('sinL.csv', sep=',')
print(data[:1])


# In[5]:


xy = np.array(data, dtype=np.float32)


# In[6]:


x_data = xy[:, 1:-1]
y_data = xy[:, [-1]]
print(len(xy),len(xy[0]))
#print(xy)
print(len(x_data),len(x_data[0]))
print(len(y_data),len(y_data[0]))
print(y_data[:5])
print(x_data[:2])
#x_data = x_data.reshape((-1,31399,50))
#y_data = y_data.reshape((-1,31399,1))


# In[7]:


X = tf.compat.v1.placeholder(tf.float32, shape=[None, 5])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) #None은 아무숫자나 가능하다는 의미
W = tf.Variable(tf.random.normal([5,1], mean=0.1014, stddev=0.1), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")
#b = tf.cast(b, dtype=tf.float64)


# In[8]:


hypothesis = 1 + tf.matmul(X, W) 


# In[210]:


cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-10)
train = optimizer.minimize(cost)


# In[216]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_, hypo_, _ = sess.run([cost,hypothesis, train] , feed_dict={X: x_data, Y: y_data})
        if step%1000==0:
            print(step,cost_,hypo_[:3])
    print("###")
    print(sess.run(W))
    w0 = sess.run(W)
    print(sess.run(hypothesis, feed_dict={X:[[2.25,5.0625,11.390625,25.62890625,57.66503906]]}))
    



