
import torch
from  torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ".format(device))


class TestCNN1(nn.Module):
    def __init__(self, input_channels, classes_num):
        super(TestCNN1, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.full_connected_1 = nn.Linear(64*7*7, 384)
        self.full_connected_2 = nn.Linear(384, classes_num)

    def forward(self, x):
        x = F.gelu(self.conv_layer_1(x))
        x = self.pool(x)
        x = F.gelu(self.conv_layer_2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.full_connected_1(x)
        x = self.full_connected_2(x)
        return x


class TestCNN2(nn.Module):
    def __init__(self, input_channels, classes_num):
        super(TestCNN2, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.batch_norm_2d = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout2d = nn.Dropout2d(0.2)
        self.fully_connected_1 = nn.Linear(64*7*7, 384)
        self.fully_connected_2 = nn.Linear(384, classes_num)
        self.batch_norm_1d = nn.BatchNorm1d(384)

    def forward(self, x):
        x = F.gelu(self.conv_layer_1(x))
        x = self.pool(x)
        x = F.gelu(self.batch_norm_2d(self.conv_layer_2(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = x.reshape(x.shape[0], -1)
        x = F.gelu(self.batch_norm_1d(self.fully_connected_1(x)))
        x = self.fully_connected_2(x)
        return x


num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10
training_pretrained_model = True


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if training_pretrained_model:
    #import sys
    my_model = torchvision.models.vgg16(pretrained=True)
    for param in my_model.parameters():
        param.requires_grad = False
    my_model.avgpool = Identity()
    my_model.features[26] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    my_model.features[28] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    my_model.classifier = nn.Sequential(
        nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, num_classes)
    )
    #print(my_model)
    #sys.exit()
    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1))
    ]), download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1))
    ]), download=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
else:

    my_model = TestCNN2(1, num_classes).to(device)

    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.00001)

my_model.to(device)

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
            #if num_images < max_images:
                # for i in range(len(predictions)):
                #     if predictions[i] != y[i] and num_images < max_images:
                #         display_image(x[i].reshape(x[0][0].shape, -1), predictions[i].item(), y[i].item())
                #         num_images += 1
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )
        return float(num_correct) / float(num_samples)



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_confusion_matrix(loader, model):
    model.eval()
    predictions = torch.empty([])
    results = torch.empty([])
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, tmp_predictions = scores.max(1)
            try:
                predictions = torch.cat((predictions, tmp_predictions))
                results = torch.cat((results, y))
            except:
                predictions = tmp_predictions
                results = y
        cm = confusion_matrix(results.cpu(), predictions.cpu())
        classes = list(range(num_classes))
        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(cm, classes)
        plt.show()



def display_image(tensor_to_display, incorrect_num, correct_num):
    img = im.fromarray((tensor_to_display.cpu().numpy()*255))
    img = img.resize((560, 560), im.LANCZOS)
    img = img.convert("RGB")
    img_d = im_draw.Draw(img)
    img_d.text((10, 10), ("Is: " + str(correct_num) + " typ: " + str(incorrect_num)), fill="#31FF2D")
    img.show()


def save_net_state(state, filename):
    print("Network state saved")
    torch.save(state, filename)


def load_net_state(state):
    print("Loading network")
    state = torch.load(state)
    my_model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


#Training & evaluate
def train_n_times(n):
    global learning_rate
    stats = pd.DataFrame(np.zeros((n, 3)), columns=['Time for epoch', 'Train accuracy', 'Test accuracy'])
    loss_stats = pd.DataFrame(np.zeros((num_epochs, 2)), columns=['Loss on epoch', 'Train time'])
    for i in range(n):
        for epoch in range(num_epochs):
            train_start_time = time.perf_counter()
            for batch_index, (data, targets) in enumerate(train_dataloader):
                data = data.to(device=device)
                targets = targets.to(device=device)
                scores = my_model(data)
                loss = criterion(scores, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Loss on epoch = {}".format(loss))
            # learning_rate = learning_rate*0.9
            # for g in optimizer.param_groups:
            #     g['lr'] = learning_rate
            train_end_time = time.perf_counter()
            loss_stats.iloc[epoch]['Loss on epoch'] = loss
            loss_stats.iloc[epoch]['Train time'] = train_end_time - train_start_time
        stats.iloc[i]['Time for epoch'] = train_end_time - train_start_time
        stats.iloc[i]['Train accuracy'] = check_accuracy(train_dataloader, my_model)
        stats.iloc[i]['Test accuracy'] = check_accuracy(test_dataloader, my_model)
        #learning_rate = 0.05
        state_to_save = {'state_dict': my_model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_net_state(state_to_save, "Results\Results_Cnn1\cnn_4_state.pth.tar")
        print(loss_stats)
        loss_stats.to_excel("Results\Results_Cnn1\cnn_4_loss_time.xlsx")
        my_model.apply(weight_reset)
    print(stats)
    #stats.to_excel("Results\Results_Cnn1\cnn_2_2.xlsx")


def load_net_to_valid():
    load_net_state("Results\Results_Cnn1\cnn_1_state.pth.tar")
    create_confusion_matrix(test_dataloader, my_model)


train_n_times(1)

#load_net_to_valid()

