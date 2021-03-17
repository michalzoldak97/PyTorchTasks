
import torch
from  torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]
import torchvision.transforms as transforms
import time

print(torch.__version__)

from math import tanh, exp
def tanhexp(x):
    return x * torch.tanh(torch.exp(x))

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, num_of_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_of_classes)
    def forward(self, x):
         #x = F.relu(self.fc1(x)) # test gelu, relu6, hardswish, mish, tanhexp
         x = F.gelu(self.fc1(x))
         x = self.fc2(x)
         return x
# # shape test
# model = FullyConnectedNN(784, 10)
# test_x = torch.rand(64, 784)
# print(model(test_x).shape)

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 28*28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10


sum_train_start_time = 0
sum_train_end_time = 0
acc_test = 0
acc_train = 0

#DataLoading
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#Network Init
my_model = FullyConnectedNN(input_size, num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_model.parameters(), lr=learning_rate)

# definition of training
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )
        return float(num_correct) / float(num_samples)

        model.train()

#reset script
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

#Network Training
for i in range(40):
    train_start_time = time.perf_counter()
    for epoch in range(num_epochs):
        for batch_index ,(data, targets) in enumerate(train_dataloader):
            #Load data to CUDA
            data = data.to(device=device)
            targets = targets.to(device=device)

            # reshape to 1 dimm
            data = data.reshape(data.shape[0], -1)

            # forward
            scores = my_model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # optimizer step
            optimizer.step()
        print("Loss on batch = {}".format(loss))
    train_end_time = time.perf_counter()
    sum_train_start_time += train_start_time
    sum_train_end_time += train_end_time
    acc_train += check_accuracy(train_dataloader, my_model)
    acc_test += check_accuracy(test_dataloader, my_model)
    my_model.apply(weight_reset)

print("Mean training time was {} train accuracy was {} test accuracy was {}".format((sum_train_end_time/40) -
                                                                                    (sum_train_start_time/40), acc_train/40,
                                                                                    acc_test/40))

