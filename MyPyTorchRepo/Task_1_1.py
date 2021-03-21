
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
import numpy as np
from PIL import Image as im
from PIL import ImageDraw as im_draw
import pandas as pd
import time


class TestCNN1(nn.Module):
    def __init__(self, input_channels, classes_num):
        super(TestCNN1, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.conv_layer_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.full_connected = nn.Linear(128*7*7, classes_num)

    def forward(self, x):
        x = F.gelu(self.conv_layer_1(x))
        x = self.pool(x)
        x = F.gelu(self.conv_layer_2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.full_connected(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ".format(device))

num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

my_model = TestCNN1(1, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_model.parameters(), lr=learning_rate) # test sgd, adaptation lr, dropout,


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    num_images = 0
    max_images = 2
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            if num_images < max_images:
                for i in range(len(predictions)):
                    if predictions[i] != y[i]:
                        display_image(x[i].reshape(x[0][0].shape, -1), predictions[i].item(), y[i].item())
                        num_images += 1
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )
        return float(num_correct) / float(num_samples)


def display_image(tensor_to_display, incorrect_num, correct_num):
    img = im.fromarray((tensor_to_display.cpu().numpy()*255))
    img = img.resize((560, 560), im.LANCZOS)
    img = img.convert("RGB")
    img_d = im_draw.Draw(img)
    img_d.text((10, 10), ("Is: " + str(correct_num) + " typ: " + str(incorrect_num)), fill="#31FF2D")
    img.show()


train_start_time = time.perf_counter()
for epoch in range(num_epochs):
    for batch_index, (data, targets) in enumerate(train_dataloader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = my_model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    print("Loss on epoch = {}".format(loss))
    train_end_time = time.perf_counter()

check_accuracy(train_dataloader, my_model)
check_accuracy(test_dataloader, my_model)