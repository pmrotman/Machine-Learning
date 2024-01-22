"""
Author: Pierce Rotman
Course: CAP 6673
Professor: Pashaie
Date: 19 September 2023

Description: Designs a single-layer perceptron to classify images of handwritten digits as either a value or not the value from 0-9.
            Checks the accuracy of the perceptron and graphs the incorrect values.
"""

#Imports
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io

#Functions needed
def sgn(x):
    """
    Calculates value of sign function for an input
    Args:
        x (float): input value
    Returns:
        (int) value of sign function
    """
    if x[0] < 0:
        return -1
    elif x[0] > 0:
        return 1
    else:
        return 0

def weight_update(X, W, e):
    """
    Updates the weights matrix
    Args:
        X (numpy.ndarray): data at current iteration
        W (numpy.ndarray): weights matrix
        e (int): error value
    Returns:
        (numpy.ndarray): updated weights matrix
    """
    return W + 0.5*(e)*X


def train(value):
    """
    Trains single layer perceptron
    Args:
        value (int): desired value for training
    Returns:
        (numpy.ndarray): updated weights matrix
    """
    initial_weights = np.zeros((784, 1))
    desired_0 = np.zeros((5000,1))
    for i in range(5000):
        if digits['trainlabels'][i][0] == value:
            desired_0[i] = 1
        else:
            desired_0[i] = -1
    last100 = []
    weights = initial_weights
    round = 0
    while sum(last100) < 100 and round < 100:
                
        for k in range(5000):
            
            input = digits['train'][:,k].reshape(784,1)
            xTw = input.transpose().dot(weights)
            y = sgn(xTw)
            desired = desired_0[k][0]
            error = desired - y
            if error == 0:
                last100.append(1)
            else:
                last100.append(0)
            if len(last100) > 100:
                del last100[0]
            weights = weight_update(input, weights, error)
            
            #print(f"{value}: {sum(last100)}")

        round += 1 
    return weights

def test(value, weights):
    """
    Tests a weights matrix with the testing data for a desired value
    Args:
        value (int): desired testing value
        weights (numpy.ndarray): weights matrix
    Returns:
        correct (int): number of correct results
        failures (dict): dictionary of frequencies of failed desired values
    """
    failures = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    desired_t_0 = np.zeros((1000, 1))
    for i in range(1000):
        if digits['testlabels'][i][0] == value:
            desired_t_0[i] = 1
        else:
            desired_t_0[i] = -1

    correct = 0
    for k in range(1000):
        input = digits['test'][:,k].reshape(784,1)
        xTw = input.transpose().dot(weights)
        y = sgn(xTw)
        desired = desired_t_0[k][0]
        error = desired - y
        if error == 0:
            correct += 1
        else:
            failures[digits['testlabels'][k][0]] += 1
    return correct, failures

digits = scipy.io.loadmat('/Users/piercerotman/Documents/MastersProgram/Machine_Learning/digits.mat')
Ws = []
Cs = []
kinds_incorrect = {}
for num in range(10):
    Ws.append(train(num))
    c, f = test(num, Ws[num])
    Cs.append(c)
    kinds_incorrect[num] = f
print([f"{val}: {Cs[val]}" for val in range(10)])
for i in range(10):
    Wim = Ws[i].reshape(28,28)
    image_test = Image.fromarray(Wim, 'L')
    image_test.save(f'{i}.png')
    #image_test.show()

df = pd.DataFrame(kinds_incorrect).T
fig, ax = plt.subplots()
df.plot(ax = ax, kind = 'bar')
plt.show()
