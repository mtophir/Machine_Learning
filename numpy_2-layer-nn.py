# 2-layer-nn supervided learning
import numpy as np

def sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input
X = np.array([[0,0,1],[0,1,0],[0,1,1],[1,0,0]])
# output
y = np.array([[1],[0],[0],[0]])

np.random.seed(1)

# randomly initialize the weight with mean 0
w0 = 2*np.random.random((3,1))-1

for i in range(1000):
    # feed forward through layer 0 and 1
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    # how much did we miss the target value?
    l1_error = y - l1
    # readjust using sigmoid
    l1_delta = l1_error*sigmoid(l1, True)
    # recalculate the weight
    w0 += np.dot(l0.T, l1_delta)

print(l1)
