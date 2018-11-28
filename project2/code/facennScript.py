'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
from scipy.optimize import minimize
from scipy.io import loadmat

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return  1.0 / (1.0 + np.exp(-z))

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #

    N = training_data.shape[0]
    
    # forwardfeed first layer
    training_data_ = np.hstack((training_data, np.zeros((training_data.shape[0], 1))))
    a = np.dot(training_data_, w1.T)
    z = sigmoid(a)
    
    # forwardfeed second layer
    z_ = np.hstack((z, np.zeros((z.shape[0], 1))))
    b = np.dot(z_, w2.T)
    o = sigmoid(b)
    
    # compute error function
    # N * 10
    label_matrix = np.zeros((o.shape[0], n_class))
    label_matrix[range(N), training_label.astype(int)] = 1
    
    obj_val = - np.sum(np.log(o) * label_matrix + (1 - label_matrix) * np.log(1 - o)) / N
    
    # add regularization
    obj_val += lambdaval / (2 * N) * (np.sum(w1 * w1) + np.sum(w2 * w2))
     
    diff = o - label_matrix
    # l * N * N * (m+1s)
    dw2 = np.dot(diff.T, z_) / N
    dw2 += (lambdaval / N) * w2
    
    # N * m
    formal12_2 = np.dot(diff, w2)[:,:-1]
    # N * m
    formal12_1 = (1.0 - z) * z
    # N * m
    formal12_1_2 = formal12_1 * formal12_2
    
    # m * N * N * (d+1)
    dw1 = np.dot(formal12_1_2.T, training_data_) / N
    dw1 += (lambdaval / N) * w1
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((dw1.flatten(), dw2.flatten()), 0)

    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    
    labels = np.array([])
    # Your code here
    
    # forwardfeed first layer
    training_data_ = np.hstack((data, np.zeros((data.shape[0], 1))))
    a = np.dot(training_data_, w1.T)
    z = sigmoid(a)
    
    # forwardfeed second layer
    z_ = np.hstack((z, np.zeros((z.shape[0], 1))))
    b = np.dot(z_, w2.T)
    o = sigmoid(b)
    
    labels = np.argmax(o, axis=1)

    return labels


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
