#3-layer-nn supervised learning using numpy
import numpy as np

def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
# input    
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# expected output (supervised learning)
y = np.array([[0],[1],[1],[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
w0 = 2*np.random.random((3,4)) - 1
w1 = 2*np.random.random((4,1)) - 1

for j in range(1000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,w0))
    l2 = nonlin(np.dot(l1,w1))
    # how much did we miss the target value?
    l2_error = y - l2        
    # readjust using sigmoid
    l2_delta = l2_error*nonlin(l2,deriv=True)
    # how much did each l1 value contribute to the l2 error?
    l1_error = l2_delta.dot(w1.T)    
    # activate the change
    l1_delta = l1_error * nonlin(l1,deriv=True)
    # recalculate the weights
    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)

print(l2)
