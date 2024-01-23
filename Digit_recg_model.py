#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install tensorflow')


# In[6]:


import tensorflow as tf
tf.version


# In[7]:


mnist=tf.keras.datasets.mnist # it is a dataset of 28x28 handwritten images 0-9


# In[8]:


(x_train,y_train), (x_test,y_test)= mnist.load_data() # loads the data in 


# In[10]:


from matplotlib import pyplot as plt


# In[11]:


plt.imshow(x_train[0])


# In[12]:


y_train[0]


# In[13]:


x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)  # scales the input data to a range 0-1


# In[22]:


model=tf.keras.models.Sequential() # adding the model
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #adding layers to the neural network
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) # relu is the basic sigmoid function
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))# 10 because final layer has 10 outputs and softmax bcz probability in final layer


# In[26]:


model.compile(optimizer='adam',  # adam is the basic out of the 10 optimizers
              loss='sparse_categorical_crossentropy',  # calculating loss using sparse instead of binary
              metrics=['accuracy'])  # taking accuracy into account while processing


# In[28]:


model.fit(x_train,y_train,epochs=3) # epochs=3 means it iterates through the data 3 times!


# In[36]:


val_loss,val_acc=model.evaluate(x_test,y_test) # loss and acuracy value


# In[37]:


model.save("mnist_dataset_reader")


# In[ ]:




