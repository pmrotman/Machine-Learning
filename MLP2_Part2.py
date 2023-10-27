"""
Author: Pierce Rotman
Course: CAP 6673
Professor: Pashaie
Date: 19 September 2023

Description: Trains a multi-layer perceptron for the identification of handwritten digits 0-9. Shows accuracy during training
and outputs incorrect values and accuracy in testing phase. 
"""

import numpy as np
import scipy.io


#Load Digits
digits = scipy.io.loadmat('/Users/piercerotman/Documents/MastersProgram/Machine_Learning/digits.mat')
np.random.seed(100)
X = np.zeros((784, 1,5000))
for k in range(5000):
    dummy = digits['train'][:,k]
    for i in range(784):
        X[i,0,k] = dummy[i]

W_inner = np.random.randn(784, 25).astype('float32')
W_outer = np.random.randn(25, 10).astype('float32')
desired = np.zeros((10, 5000), dtype='float32')
for k in range(5000):
    val = digits['trainlabels'][k]
    desired[val, k] = 1
print(digits['trainlabels'].shape)
def sigmoid(data):
    """
    Calculate sigmoid function for data
    Args:
        data (numpy.ndarray): input data
    Returns:
        sigmoid (numpy.ndarray):
    """
    return 1/(1+np.exp(-data*0.5))

def sigmoid_prime(data):
    """
    Calculate sigmoid derivative for data
    Args: 
        data (numpy.ndarray): input data
    Returns:
        sigmoid prime (numpy.ndarray): derivative of sigmoid for data
    """
    return sigmoid(data) * (1-sigmoid(data*0.5))


#Train
alpha = 0.005
mc = 1

overall_max = {'Round': 0, 'Alpha': 0, 'MC': 0, 'Correct': 0}
last_loss = 0
for run in range(100):
    loss = 0
    correct = 0
    for k in range(5000):
        #Forward
        data = X[:,:,k]

        inner_vector = data.transpose().dot(W_inner)
        inner_output = sigmoid(inner_vector)
        outer_vector = inner_output.dot(W_outer)
        output = sigmoid(outer_vector)

        #Error Calculation
        error = desired[:,k] - output

        #Loss calculation
        loss += 0.5 * np.sum(error ** 2)
        val = digits['trainlabels'][k][0]
        if np.argmax(output) == digits['trainlabels'][k][0]:
            correct += 1
        
        #Back propagation
        outer_delta = error * sigmoid_prime(outer_vector)
        inner_delta = sigmoid_prime(inner_vector) * (outer_delta.dot(W_outer.transpose()))


        #Update weights:
        W_outer = mc*W_outer + alpha * inner_output.transpose().dot(outer_delta)
        W_inner = mc*W_inner + alpha * data.dot(inner_delta)
    
    if (loss-last_loss)**2 < 0.00001:
        alpha = 0.01
    alpha = alpha / (1 + alpha * 0.00001)
    print(f"Round: {run}, Loss: {loss}, Total Correct: {correct}")
    if correct >= 4800:
        break
    last_loss = loss

#test
Xtest = np.zeros((784,1,1000))
for k in range(1000):
    dummy = digits['test'][:,k]
    for i in range(784):
        Xtest[i,0,k] = dummy[i]

correct = 0
kind_incorrect = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

correct_predictions = 0
kind_incorrect2 = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
for k in range(1000):
    #Forward
    data = Xtest[:,:,k]
    inner_vector = data.transpose().dot(W_inner)
    inner_output = sigmoid(inner_vector)
    outer_vector = inner_output.dot(W_outer)
    y = sigmoid(outer_vector)

    if np.argmax(y) == digits['testlabels'][k][0]:
        correct += 1
    else:
        kind_incorrect[digits['testlabels'][k][0]] += 1

print(f"{100*correct/1000}%") 
print(kind_incorrect)
    
