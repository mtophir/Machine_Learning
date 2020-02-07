#2-layer-nn using pytorch

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

w0 = 2*torch.randn((3,1))-1

for i in range(1000):
    l0 = X
    l1 = sigmoid(torch.mm(l0, w0))
    l1_error = y - l1
    l1_delta = l1_error*sigmoid(l1, deriv=True)
    
    w0 += torch.mm(l0.T, l1_delta)
    
print(l1)
