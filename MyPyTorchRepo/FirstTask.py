#Check if available

import torch

# tensor initialazing

device = "cuda" if torch.cuda.is_available() else "cpu"

test_tensor = torch.tensor([[1,2,3],[3,2,1]])
test_tensor = torch.tensor([[1,2,3],[3,2,1]], dtype=torch.float16)
test_tensor = torch.tensor([[1,2,3],[3,2,1]], dtype=torch.float16, device="cuda")
test_tensor = torch.tensor([[1,2,3],[3,2,1]], dtype=torch.float16, device=device, requires_grad=True)

test_tensor = torch.empty(size=(3,3,3))
test_tensor = torch.zeros((2,3))
test_tensor = torch.ones((2,3))
test_tensor = torch.rand((2,3)) # >0<1
test_tensor = torch.eye(3,3)
test_tensor = torch.arange(0,10,1)
test_tensor = torch.linspace(0.1, 5, 20)
test_tensor = torch.empty((2,3)).normal_(0,1)
test_tensor = torch.empty((2,3)).uniform_(0,1)
test_tensor = torch.diag(torch.rand(3))

# tensor dtype convertion
test_tensor = test_tensor.bool()
test_tensor = torch.tensor([[1,2,3], [1,2,3]], device=device)
test_tensor = test_tensor.long()
test_tensor = test_tensor.half()
test_tensor = test_tensor.double()

# Tensor to np.array
import numpy as np
test_tensor = np.zeros((12,12))
test_tensor = torch.from_numpy(test_tensor)
test_array = test_tensor.numpy()
#print(test_tensor, "\n" , test_tensor.dtype, "\n" , test_tensor.device)
#print(test_tensor.shape)

# Tensor math operations

x = torch.ones((2,3))
y = x*3
y = x+y
z = torch.zeros((2,3))
y = torch.add(x,y,out=z)

z = y-x
z = torch.true_divide(x,y)

# inplace operations
x.add_(y)
z += x
z = z ** 2

print(y,z)