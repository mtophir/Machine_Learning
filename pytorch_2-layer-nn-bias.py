import torch

def sigmoid(x, deriv=False):
   if (deriv==True):
       return x * (1-x)
   return 1/(1+torch.exp(-x))

#Input tensor
X = torch.Tensor([[1,0,1,0],[1,0,1,1],[0,1,0,1],[0,1,0,0]])
#Output
y = torch.Tensor([[1],[1],[0],[0]])

#Variable initialization
#setting training iterations
epoch=1000
#setting learning rate
lr=0.1
#dimension or number of features in data set
D = X.shape[1]
#number of hidden layer neurons
H = 4
#number of neurons in output layer
OUT = 1

#weight initialization
w0=torch.randn(D, H).type(torch.FloatTensor)
w1=torch.randn(H, OUT)

#bias initialization
b0=torch.randn(1, H).type(torch.FloatTensor)
b1=torch.randn(1, OUT)

for i in range(epoch):
    
    #Forward Propogation
    
    #activate of hidden layer
    l1 = torch.mm(X, w0) + b0
    l1_grad = sigmoid(l1)
    
    #activation of output layer
    l2 = torch.mm(l1_grad, w1) + b1
    output = sigmoid(l2)

    #loss calculation
    l2_error = y - output
    
    #Back Propagation
    
    #compute derivative of error terms
    l2_act = sigmoid(output, True)
    l1_act = sigmoid(l1_grad, True)
    
    #backpass the changes to previous layers
    l2_delta = l2_error * l2_act
    l1_error = torch.mm(l2_delta, w1.t())
    l1_delta = l1_error * l1_act
    
    #recalculate weights and bias
    w1 += torch.mm(l1_grad.t(), l2_delta) *lr
    b1 += l2_delta.sum() *lr
    w0 += torch.mm(X.t(), l1_delta) *lr
    b0 += l2_delta.sum() *lr

print('actual :\n', y, '\n')
print('predicted :\n', output)
