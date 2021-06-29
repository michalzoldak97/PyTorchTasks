
import torch
from  torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image as im
from PIL import ImageDraw as im_draw
import pandas as pd
import time
import matplotlib.pyplot as plt
from ImgLoader import ImgLoader

num_classes = 2
learning_rate = 0.001
batch_size = 128
num_epochs = 300

l1_weight = 0.0001

train_set = ImgLoader(csv_file='DATA_256/images_train.csv', root_dir='', transform=transforms.ToTensor())
test_set = ImgLoader(csv_file='DATA_256/images_test.csv', root_dir='', transform=transforms.ToTensor())
val_set = ImgLoader(csv_file='DATA_256/images_val.csv', root_dir='', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)


class TestCNN3(nn.Module):
    def __init__(self, input_channels, classes_num):
        super(TestCNN3, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.conv_layer_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.conv_layer_3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.conv_layer_4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.batch_norm_2d_1 = nn.BatchNorm2d(8)
        self.batch_norm_2d_2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout2d = nn.Dropout2d(0.2)
        self.fully_connected_1 = nn.Linear(4096, 16)
        self.fully_connected_2 = nn.Linear(16, num_classes)
        self.dropout1d = nn.Dropout(0.2)
        self.batch_norm_1d_1 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.gelu(self.batch_norm_2d_1(self.conv_layer_1(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = F.gelu(self.batch_norm_2d_1(self.conv_layer_2(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = F.gelu(self.batch_norm_2d_2(self.conv_layer_3(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = F.gelu(self.batch_norm_2d_2(self.conv_layer_4(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = x.reshape(x.shape[0], -1)
        # print("X shape: {}".format(x.shape))
        x = F.gelu(self.batch_norm_1d_1(self.fully_connected_1(x)))
        x = self.dropout1d(x)
        x = self.fully_connected_2(x)
        return x

    def init_layers(self):
        nn.init.xavier_uniform_(self.conv_layer_1.weight)
        nn.init.constant_(self.conv_layer_1.bias, .00000001)
        nn.init.xavier_uniform_(self.conv_layer_2.weight)
        nn.init.constant_(self.conv_layer_2.bias, .00000001)
        nn.init.xavier_uniform_(self.conv_layer_3.weight)
        nn.init.constant_(self.conv_layer_3.bias, .00000001)
        nn.init.xavier_uniform_(self.conv_layer_4.weight)
        nn.init.constant_(self.conv_layer_4.bias, .00000001)
        nn.init.xavier_uniform_(self.fully_connected_1.weight)
        nn.init.constant_(self.fully_connected_1.bias, .00000001)
        nn.init.xavier_uniform_(self.fully_connected_2.weight)
        nn.init.constant_(self.fully_connected_2.bias, .00000001)


class TestCNN4(nn.Module):
    def __init__(self, input_channels, classes_num):
        super(TestCNN4, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.conv_layer_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.conv_layer_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.conv_layer_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))
        self.batch_norm_2d_1 = nn.BatchNorm2d(16)
        self.batch_norm_2d_2 = nn.BatchNorm2d(16)
        self.batch_norm_2d_3 = nn.BatchNorm2d(32)
        self.batch_norm_2d_4 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout2d = nn.Dropout2d(0.2)
        self.fully_connected_1 = nn.Linear(8192, 16)
        self.fully_connected_2 = nn.Linear(16, num_classes)
        self.dropout1d = nn.Dropout(0.2)
        self.batch_norm_1d_1 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.gelu(self.batch_norm_2d_1(self.conv_layer_1(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = F.gelu(self.batch_norm_2d_2(self.conv_layer_2(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = F.gelu(self.batch_norm_2d_3(self.conv_layer_3(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = F.gelu(self.batch_norm_2d_4(self.conv_layer_4(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = x.reshape(x.shape[0], -1)
        # print("X shape: {}".format(x.shape))
        x = F.gelu(self.batch_norm_1d_1(self.fully_connected_1(x)))
        x = self.dropout1d(x)
        x = self.fully_connected_2(x)
        return x

    def init_layers(self):
        nn.init.xavier_uniform_(self.conv_layer_1.weight)
        nn.init.constant_(self.conv_layer_1.bias, .00000001)
        nn.init.xavier_uniform_(self.conv_layer_2.weight)
        nn.init.constant_(self.conv_layer_2.bias, .00000001)
        nn.init.xavier_uniform_(self.conv_layer_3.weight)
        nn.init.constant_(self.conv_layer_3.bias, .00000001)
        nn.init.xavier_uniform_(self.conv_layer_4.weight)
        nn.init.constant_(self.conv_layer_4.bias, .00000001)
        nn.init.xavier_uniform_(self.fully_connected_1.weight)
        nn.init.constant_(self.fully_connected_1.bias, .00000001)
        nn.init.xavier_uniform_(self.fully_connected_2.weight)
        nn.init.constant_(self.fully_connected_2.bias, .00000001)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_model = TestCNN4(3, num_classes)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(my_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.00001)
#optimizer = optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=0.001)
optimizer = optim.AdamW(my_model.parameters(), lr=learning_rate, weight_decay=0.01, amsgrad=True)

my_model.to(device)


def check_accuracy(loader, model):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )
        return float(num_correct) / float(num_samples)


def calculate_lr(curr_lr, past_result, curr_result):
    if curr_lr > 0.00005:
        if curr_result/past_result < .5:
            return curr_lr
        elif curr_result/past_result < .75 and curr_lr * .9 > 0.00005:
            return curr_lr * .95
        elif curr_result/past_result < .9 and curr_lr * .8 > 0.00005:
            return curr_lr * .85
        elif curr_lr * .5 > 0.00005:
            return curr_lr * .75
        else:
            return curr_lr
    else:
        return curr_lr


def optimize_lr(past_result, curr_result):
    global learning_rate
    learning_rate = calculate_lr(learning_rate, past_result, curr_result)
    for g in optimizer.param_groups:
        g['lr'] = learning_rate


def increase_lr():
    global learning_rate
    if learning_rate <= 0.00008:
        learning_rate = 0.00025#learning_rate * 2
        for g in optimizer.param_groups:
            g['lr'] = learning_rate


def save_net_state(state, filename):
    torch.save(state, filename)

max_acc = 0
def set_max_acc(curr_acc):
    global max_acc
    if curr_acc > max_acc:
        max_acc = curr_acc
        return True
    else:
        return False


def train():
    global max_acc
    stag_max_train = 0
    checkpoint_num = 40
    stats = pd.DataFrame(np.zeros((int(num_epochs/checkpoint_num), 3)), columns=['Loss on epoch', 'Train accuracy', 'Test accuracy'])
    loss_stats = pd.DataFrame(np.zeros((num_epochs, 2)), columns=['Loss on epoch', 'Train time'])
    my_model.init_layers()
    my_model.train()
    for epoch in range(num_epochs):
        train_start_time = time.perf_counter()
        for batch_index, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = my_model(data)
            loss = criterion(scores, targets)
            #l1_penalty = l1_weight * sum([p.abs().sum() for p in my_model.parameters()])
            #loss_penalty = loss + l1_penalty
            optimizer.zero_grad()
            loss.backward() #changed from loss.backward
            optimizer.step()
        train_end_time = time.perf_counter()
        loss_stats.iloc[epoch]['Loss on epoch'] = loss
        loss_stats.iloc[epoch]['Train time'] = train_end_time - train_start_time
        if epoch > 0:
            optimize_lr(loss_stats.iloc[epoch-1]['Loss on epoch'], loss_stats.iloc[epoch]['Loss on epoch'])

        # net state manipulation
        if epoch > 0 and epoch % checkpoint_num == 0:
            curr_loc = int(epoch/checkpoint_num)
            stats.iloc[curr_loc]['Loss on epoch'] = loss
            stats.iloc[curr_loc]['Train accuracy'] = check_accuracy(train_loader, my_model)
            stats.iloc[curr_loc]['Test accuracy'] = check_accuracy(test_loader, my_model)
            stats.to_csv('RESULTS/stats_net1_5.csv')
            if curr_loc > 0:
                if set_max_acc(stats.iloc[curr_loc]['Test accuracy']):
                    state_to_save = {'state_dict': my_model.state_dict(), 'optimizer': optimizer.state_dict()}
                    save_net_state(state_to_save, "NETWORKS/test5.pth.tar")
                else:
                    load_net_state("NETWORKS/test5.pth.tar")
                    print("Best Test acc was {}, is now {}, so reloading network".format(max_acc, stats.iloc[curr_loc]['Test accuracy']))
                    increase_lr()
                    if stats.iloc[curr_loc]['Train accuracy'] > 90:
                        stag_max_train += 1
                    if stag_max_train > 9:
                        print("Train max reached")
                        break
                    elif stats.iloc[curr_loc]['Train accuracy'] > 99.999:
                        stag_max_train += 1

        loss_stats.to_csv('RESULTS/loos_net1_5.csv')
        print("Loss on epoch = {}  epoch num: {}  learning rate: {}".format(loss, epoch, learning_rate))

    loss_stats.to_csv('RESULTS/loos_net1_5.csv')
    stats.to_csv('RESULTS/stats_net1_5.csv')
    check_accuracy(train_loader, my_model)
    check_accuracy(test_loader, my_model)
    check_accuracy(val_loader, my_model)


def load_net_state(state):
    print("Loading network")
    state = torch.load(state)
    my_model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


def load_net_to_valid():
    load_net_state("NETWORKS/test3_G.pth.tar")
    check_accuracy(train_loader, my_model)
    check_accuracy(test_loader, my_model)
    check_accuracy(val_loader, my_model)


# train()
load_net_to_valid()
