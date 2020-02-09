#3-layer-nn using pytorch

import torch

def sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+torch.exp(-x))

#input tensor
X = torch.Tensor([[0,0,1],[0,1,0],[0,1,1],[1,0,0]])
#expected output (supervised learning)
y = torch.Tensor([[0],[1],[1],[0]])

torch.manual_seed(1)

# randomly initialize our weights with mean 0
w0 = 2*torch.randn((3,4))-1
w1 = 2*torch.randn((4,1))-1

for i in range(1000):
    
    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(torch.mm(l0, w0))
    l2 = sigmoid(torch.mm(l1, w1))
    # how much did we miss the target value?
    l2_error = y - l2
    # readjust using sigmoid
    l2_delta = l2_error*sigmoid(l2, deriv=True)
    # how much did each l1 value contribute to the l2 error?
    l1_error = l2_delta.mm(w1.T)
    # activate the change
    l1_delta = l1_error*sigmoid(l1, deriv=True)
    # recalculate the weights
    w1 += torch.mm(l1.T, l2_delta)
    w0 += torch.mm(l0.T, l1_delta)
    
print(l2)
