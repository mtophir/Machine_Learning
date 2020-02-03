import numpy as np

def sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0,0,1],[0,1,0],[0,1,1],[1,0,0]])
y = np.array([[1],[0],[0],[0]])

np.random.seed(1)

w0 = 2*np.random.random((3,1))-1

for i in range(1000):
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l1_error = y - l1
    l1_delta = l1_error*sigmoid(l1, True)
    
    w0 += np.dot(l0.T, l1_delta)

print(l1)
