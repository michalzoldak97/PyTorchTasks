# Check if available

import torch

# tensor initialazing

device = "cuda" if torch.cuda.is_available() else "cpu"
#
# test_tensor = torch.tensor([[1,2,3],[3,2,1]])
# test_tensor = torch.tensor([[1,2,3],[3,2,1]], dtype=torch.float16)
# test_tensor = torch.tensor([[1,2,3],[3,2,1]], dtype=torch.float16, device="cuda")
# test_tensor = torch.tensor([[1,2,3],[3,2,1]], dtype=torch.float16, device=device, requires_grad=True)
#
# test_tensor = torch.empty(size=(3,3,3))
# test_tensor = torch.zeros((2,3))
# test_tensor = torch.ones((2,3))
# test_tensor = torch.rand((2,3)) # >0<1
# test_tensor = torch.eye(3,3)
# test_tensor = torch.arange(0,10,1)
# test_tensor = torch.linspace(0.1, 5, 20)
# test_tensor = torch.empty((2,3)).normal_(0,1)
# test_tensor = torch.empty((2,3)).uniform_(0,1)
# test_tensor = torch.diag(torch.rand(3))
#
# # tensor dtype convertion
# test_tensor = test_tensor.bool()
# test_tensor = torch.tensor([[1,2,3], [1,2,3]], device=device)
# test_tensor = test_tensor.long()
# test_tensor = test_tensor.half()
# test_tensor = test_tensor.double()

# Tensor to np.array
import numpy as np

# test_tensor = np.zeros((12,12))
# test_tensor = torch.from_numpy(test_tensor)
# test_array = test_tensor.numpy()
# #print(test_tensor, "\n" , test_tensor.dtype, "\n" , test_tensor.device)
# #print(test_tensor.shape)
#
# # Tensor math operations
#
# x = torch.ones((2,3))
# y = x*3
# y = x+y
# z = torch.zeros((2,3))
# y = torch.add(x,y,out=z)
#
# z = y-x
# z = torch.true_divide(x,y)
#
# # inplace operations
# x.add_(y)
# z += x
# z = z ** 2
#
# # matrix multiplication
# x1 = torch.rand(2,5)
# x2 = torch.rand(5,3)
# x3 = torch.mm(x1, x2)
# # matrinx exponentation
# sth_matrix = torch.rand(2,2)
# sth_matrix.matrix_power(2)
# # elementwise multiplication
# z = x * y
# # dot product
# z = torch.dot(x, y)
# #bath matrix multiplication
# batch = 32
# n = 10
# m = 20
# p = 30
# tensor1 = torch.rand(batch, n, m)
# tensor2 = torch.rand(batch, m, p)
# out_bmm = torch.bmm(tensor1, tensor2) # result will be (batch, n, p)
# #broadcasting
# x1 = torch.rand((5, 5))
# x2 = torch.rand((1, 5))
# z = x1 - x2
# z = x1 ** x2
# sum_x1 = torch.sum(x1, dim=1)
# max_x1_values, max_x1_indices = torch.max(x1, dim=1)
# abs_x1 = torch.abs(x1)
# argmax_x1 = torch.argmax(x1, dim=1)  # indexes of max values
# argmin_x1 = torch.argmin(x1, dim=1)  # indexes of min values
# mean_x1 = torch.mean(x1.float(), dim=1)
# equality_x1_x2 = torch.eq(x1, x2)
# sorted_x1 = torch.sort(x1, descending=True)
# clamped_x1 = torch.clamp(x1, min=0)
# bools_tensor = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
# b_t_result = torch.any(bools_tensor)
# b_t_result = torch.all(bools_tensor)

######## Tensor Indexing ###########
#
# batch_size = 5
# features = 3
# x1 = torch.rand((batch_size, features))
# print("Test matrix 1: {} \n Test matrix shape: {} \n features {}".format(x1, x1.shape, x1[0].shape))
# x1_fragment = x1[4, :2]
# indices = [0,3,4]
# # print(x1[indices])
# row_x1 = torch.tensor([0,0]) #rownum
# col_x1 = torch.tensor([2,0]) #colnum
# x1 = torch.arange(10)
# z = x1[(x1 < 2) | (x1 > 8)]
# evenelements = x1[x1.remainder(2) == 0]
#
# queryTorch = torch.where(x1 > 5, x1, x1 + 3)
# unique_val = torch.tensor([1, 2, 3, 2, 1, 1, 1133, 3454, 22, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 2, 0, 3, 2, 3, 2]).unique()
# fd_tensor = torch.zeros((2, 2, 2, 2))
# fd_tensor = fd_tensor.numel()
#
# x1 = torch.arange(9)
# x2 = x1.view(3, 3) #faster but requires contiquesl
# x2 = x1.reshape(3,3)
#
# x1 = torch.rand((3,4))
# x2 = torch.rand((3,4))
# xcat = torch.cat((x1,x2), dim=0) #concatenate
# xcat_1 = torch.cat((x1,x2), dim=1)
# unroll = x1.view(-1)
#
#
# import pandas as pd
# df = pd.DataFrame([0], columns=['R'])
# print(df)
from PIL import Image, ImageDraw

test_img = torch.rand((28, 28))
test_img = test_img.cpu().numpy() * 255
test_img = Image.fromarray(test_img)
test_img = test_img.resize((280, 280), Image.LANCZOS)
test_img = test_img.convert("RGB")
test_draw = ImageDraw.Draw(test_img)
test_draw.text((10, 10), str(10),  fill="#31FF2D")
test_img.show()
print(test_img)