#4-layer-nn using pytorch

import torch

def sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+torch.exp(-x))

#input tensor
X = torch.Tensor([[0,0,1],[0,1,0],[0,1,1],[1,0,0]])
#expected output (supervised learning)
y = torch.Tensor([[1],[1],[0],[0]])

torch.manual_seed(1)

# randomly initialize our weights with mean 0
w0 = 2*torch.randn((3,4))-1
w1 = 2*torch.randn((4,4))-1
w2 = 2*torch.randn((4,1))-1

for i in range(1000):
    
    # Feed forward through layers 0, 1, 2 and 3
    l0 = X
    l1 = sigmoid(torch.mm(l0, w0))
    l2 = sigmoid(torch.mm(l1, w1))
    l3 = sigmoid(torch.mm(l2, w2))
    # how much did we miss the target value?
    l3_error = y - l3
    l3_delta = l3_error*sigmoid(l3, deriv=True)
    # how much did each l2 value contribute to the l3 error?
    l2_error = l3_delta.mm(w2.T)
    l2_delta = l2_error*sigmoid(l2, deriv=True)
    # how much did each l1 value contribute to the l2 error?
    l1_error = l2_delta.mm(w1.T)
    l1_delta = l1_error*sigmoid(l1, deriv=True)    
    # recalculate the weights
    w2 += torch.mm(l2.T, l3_delta)
    w1 += torch.mm(l1.T, l2_delta)
    w0 += torch.mm(l0.T, l1_delta)
    
print(l3)
