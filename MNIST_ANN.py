
# coding: utf-8

# In[3]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[4]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


X_train= mnist.train.images[:,:]
X_test = mnist.test.images[:,:]
y_train = mnist.train.labels
y_test = mnist.test.labels


# In[20]:


print(X_train.shape)
print(y_test.shape)


# In[7]:


image = X_train[530].reshape([28,28])
plt.imshow(image, cmap=plt.cm.binary)#=plt.get_cmap('viridis'))
plt.show()


# In[8]:


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1


# In[9]:


# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


# In[10]:


# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])


# In[11]:


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[12]:


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer



# In[13]:


# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()


# In[14]:


sess = tf.Session()


# In[15]:



sess.run(init)

    # Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
        avg_cost += c / total_batch
        # Display logs per epoch step
    if epoch % display_step == 0:
            #print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        print("Epoch:", '%04d' % (epoch+1), "cost=",'%.9f' %(avg_cost))

print("Optimization Finished!")

    # Test model
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy:", sess.run(accuracy, feed_dict={X: X_test, Y: y_test}))
    
    


# In[30]:


def display_compare(num):
    x_test= mnist.test.images[num,:].reshape(1,784)
    y_test = mnist.test.labels[num,:]

    label = y_test.argmax()
    # THIS GETS OUR PREDICTION AS A INTEGER
    prediction = sess.run(pred, feed_dict={X: x_test}).argmax()
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(x_test.reshape([28,28]), cmap=plt.cm.binary)# plt.get_cmap('gray_r'))
    plt.show()


# In[49]:


import random as ran
display_compare(ran.randint(0, 10000))

