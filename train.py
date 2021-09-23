# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:43:46 2021

@author: Momo
"""
#https://github.com/zalandoresearch/fashion-mnist
#https://datascience-enthusiast.com/DL/Tensorflow_Tutorial.html
import tensorflow as tf
import math
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from six.moves import cPickle 
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
def create_one_hot(data):
  
  shape = (data.size, data.max()+1)
  one_hot = np.zeros(shape)
  rows = np.arange(data.size)
  one_hot[rows, data] = 1

  return one_hot


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(dtype="float",shape=[n_x,None])
    Y = tf.placeholder(dtype="float",shape=[n_y,None])
    ### END CODE HERE ###
    
    return X, Y

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [128, 784]
                        b1 : [128, 1]
                        W3 : [10, 128]
                        b3 : [10, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [128,784], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [128,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [10,128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [10,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters



def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                             # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2

    ### END CODE HERE ###
    
    return Z2

def compute_cost(Z2, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (10, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)
    
    
    y_hat_softmax  = tf.nn.softmax(logits)
    #cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y_hat_softmax + 1e-10), [1]))

    cost = -tf.reduce_sum(labels*tf.log(y_hat_softmax + 1e-10))


    ### START CODE HERE ### (1 line of code)
    #cost = -tf.reduce_sum(labels*tf.log(logits + 1e-10))  #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = labels))
    ### END CODE HERE ###
    
    return cost

def random_mini_batches(X, Y, mini_batch_size=32):
    """
    Arguments:
    X = input data which have n features and m samples
    Y = output label 
    mini_batch_size = a hyperparameters
    Returns:
    mini_batches = a list contains each mini batch as [(mini_batch_X1, mini_batch_Y1), (mini_batch_X2, minibatch_Y2),....]
    """
  

    m = X.shape[1]
    mini_batches = []
  
    permutation = list(np.random.permutation(m)) # transform a array into a list containing ramdom index
    X_shuffled = X[:, permutation] # shuffle X randomly 
    Y_shuffled = Y[:, permutation].reshape((10, m))
  
    num_batches = int(m/mini_batch_size)
  
    for i in range(num_batches):
        mini_batch_X = X_shuffled[:, i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_Y = Y_shuffled[:, i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m/mini_batch_size != 0:
        mini_batch_X = X_shuffled[:, num_batches*mini_batch_size:m]
        mini_batch_Y = Y_shuffled[:, num_batches*mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches





def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.000001,
          num_epochs = 1500, minibatch_size = 60, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 784, number of training examples = 60000)
    Y_train -- test set, of shape (output size = 10, number of training examples = 60000)
    X_test -- training set, of shape (input size = 784, number of training examples = 10000)
    Y_test -- test set, of shape (output size = 10, number of test examples = 10000)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y =  create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)       
    
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z2 =forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost =compute_cost(Z2, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
     
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1


            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)


            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                #print ("Cost in epoch %i: %f" % (epoch, minibatch_cost))
                epoch_cost += minibatch_cost 
                
            epoch_cost = epoch_cost/ num_minibatches
            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        var_list = [v for v in tf.trainable_variables()]
        val = sess.run(var_list)
        NUMPY_WEIGHT = r'C:\Users\Momo\Documents\yosinski\param.pkl'
        with open(NUMPY_WEIGHT,'wb') as fid:
            cPickle.dump(val,fid,protocol=cPickle.HIGHEST_PROTOCOL)
            
            
        return parameters


def model_load_param(learning_rate = 0.0001,
          num_epochs = 100, print_cost = True):

    costs =[]
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    
    image = np.random.uniform(size=(28,28))
    #image = image_ori / 255.0
    image = np.reshape(image, (1, -1))
    image = np.rollaxis(image, 1, 0)
    
    GT  = np.zeros(10).reshape(10,1)
    GT[0,0] = 1
    (n_x, m) = image.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = GT.shape[0]    


    X, Y =  create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z2 =forward_propagation(X, parameters)
    ### END CODE HERE ###
    loss = Z2[4,0]  #size(10,?)  
    gradient = tf.gradients(loss, X)
    

    #optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        ########### load and asign param ############
        NUMPY_WEIGHT = r'C:\Users\Momo\Documents\yosinski\param.pkl'
        with open(NUMPY_WEIGHT,'rb') as fid:
           val = cPickle.load(fid)          
        var_list = [v for v in tf.trainable_variables()]
        custom_load_ops = []    
        for var,v in zip(var_list[0:],val):
           custom_load_ops.append(tf.assign(var,v))

        sess.run(custom_load_ops)

        ###########################################
        # Do the training loop
        for epoch in range(num_epochs):
            
            
             grad, loss_value = sess.run([gradient, loss], feed_dict={X: image, Y: GT})
            
            
             correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))
             accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
             print ("Class Pred:", accuracy.eval({X: image, Y: GT}))
            
             grad = np.array(grad).squeeze(axis=0)
             step_size = 1.0 / (grad.std() + 1e-8)
             image += step_size * grad
             image = np.clip(image, 0.0, 1.0)

         

             # Print the cost every epoch
             if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, loss_value))
             if print_cost == True and epoch % 5 == 0:
                img = np.round(np.reshape(255*image,(28,28))).astype(np.uint8)
                plt.imshow(img, "gray")
                #img = ndimage.gaussian_filter(img, sigma=(1, 0 ), order=0)
                #plt.imshow(img, "gray", interpolation='nearest', )

                plt.show()
                costs.append(loss_value)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
           
        return parameters
    
    
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

temp = np.reshape(train_images, (60000, -1))
X_train = np.rollaxis(temp, 1, 0)

temp = np.reshape(test_images, (10000, -1))
X_test =np.rollaxis(temp, 1, 0)

temp = create_one_hot(train_labels)
Y_train =np.rollaxis(temp, 1, 0)

temp = create_one_hot(test_labels)
Y_test =np.rollaxis(temp, 1, 0)

X, Y = create_placeholders(784, 10)

# Model training
#parameters = model(X_train, Y_train, X_test, Y_test)

#Fature optimization
parameters = model_load_param()

