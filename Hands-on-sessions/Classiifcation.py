
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


x1_label0= np.random.normal(1,1,[100,1])
x2_label0=np.random.normal(1,1,[100,1])
test_x1_label0= np.random.normal(1,2,[30,1])
test_x2_label0=np.random.normal(1,2,[30,1])


# In[3]:


x1_label1= np.random.normal(5,1,[100,1])
x2_label1=np.random.normal(4,1,[100,1])
test_x1_label1= np.random.normal(5,2,[30,1])
test_x2_label1=np.random.normal(4,2,[30,1])

x1_label2= np.random.normal(8,1,[100,1])
x2_label2=np.random.normal(0,1,[100,1])
test_x1_label2= np.random.normal(8,2,[30,1])
test_x2_label2=np.random.normal(0,2,[30,1])


# In[4]:


plt.scatter(test_x1_label0,test_x2_label0,c="r",marker="o",s=10)
plt.scatter(test_x1_label1,test_x2_label1,c="g",marker="x",s=10)
plt.scatter(test_x1_label2,test_x2_label2,c="b",marker="+",s=10)
plt.show()


# In[5]:


xs_label0= np.hstack((x1_label0,x2_label0))
xs_label1= np.hstack((x1_label1,x2_label1))
xs_label2= np.hstack((x1_label2,x2_label2))
xs= np.vstack((xs_label0,xs_label1,xs_label2))
print(xs.shape)

t_xs_label0= np.hstack((test_x1_label0,test_x2_label0))
t_xs_label1= np.hstack((test_x1_label1,test_x2_label1))
t_xs_label2= np.hstack((test_x1_label2,test_x2_label2))
t_xs= np.vstack((t_xs_label0,t_xs_label1,t_xs_label2))
print(len(t_xs))


# In[6]:


y=np.matrix([[1,0,0]]*len(xs_label0)+[[0,1,0]]*len(xs_label1)+[[0,0,1]]*len(xs_label2))
t_y=np.matrix([[1,0,0]]*len(t_xs_label0)+[[0,1,0]]*len(t_xs_label1)+[[0,0,1]]*len(t_xs_label2))


# In[7]:


#print(y)
print(t_y)


# In[8]:


arr=np.arange(len(xs))
np.random.shuffle(arr)

t_arr=np.arange(len(t_xs))
np.random.shuffle(t_arr)


# In[9]:


xs=xs[arr,:]
y= y[arr,:]
print(y.shape)

t_xs=t_xs[t_arr,:]
t_y= t_y[t_arr,:]
print(t_y.shape)


# In[10]:


train_size, num_features= xs.shape


# In[11]:


print(num_features)
print(xs.shape)


# In[24]:


l_rate=0.01
nepochs=1000
batch_size=100
num_labels=3


# In[25]:


w= tf.Variable(tf.zeros(shape=[num_features,num_labels]))
b= tf.Variable(tf.zeros(shape=[num_labels]))
X= tf.placeholder("float",shape= [None,num_features])
Y= tf.placeholder("float",shape= [None,num_labels])


# In[26]:


y_model= tf.nn.softmax(tf.matmul(X,w)+b)
cost= -tf.reduce_sum(Y*tf.log(y_model))
train_op= tf.train.GradientDescentOptimizer(l_rate).minimize(cost)


# In[27]:


corr_pred= tf.equal(tf.argmax(y_model,1),tf.argmax(Y,1))
accuracy= tf.reduce_mean(tf.cast(corr_pred,"float"))


# In[28]:


with tf.Session() as sess:
    init=tf.global_variables_initializer()
    init.run()
    for step in range (nepochs*train_size//batch_size):
        offset= (step*batch_size)%train_size
        batch_xs= xs[offset:offset+batch_size,:]
        batch_y= y[offset:offset+batch_size,:]
        err,_= sess.run([cost,train_op], feed_dict={X: batch_xs, Y: batch_y})
        #y_m= sess.run(y_model,feed_dict={X: batch_xs, Y: batch_y} )
        #print(y_m)
        if step%100==0:
            #plt.scatter(step,err)
             print(step,err)
            
    w_val = sess.run(w)
    b_val = sess.run(b)
    #print('learned parameters', w_val)
    #print('learned parameters-b', b_val)
    #plt.show()
    acc= sess.run(accuracy, feed_dict={X: t_xs, Y: t_y})
    print(acc)


    

