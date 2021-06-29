from torch_geometric.datasets import MNISTSuperpixels
import torch
from torch_geometric.nn.models import GraphUNet
import torch.nn.functional as F
from torch_geometric.nn import  FiLMConv, ResGatedGraphConv
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import pandas as pd
import os
import sys
import time

dataset = MNISTSuperpixels(root='/tmp/MNIST/test', train=False)

train_dataset = MNISTSuperpixels(root='/tmp/MNIST/train', train=True)
test_dataset = MNISTSuperpixels(root='/tmp/MNIST/test', train=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

learning_rate = .01
max_result = 0
test_failed = 0
num_before_imp = 0


def optimize_lr():
    global learning_rate
    if learning_rate * .99 > .0005:
        learning_rate = learning_rate * .99
        for g in optimizer.param_groups:
            g['lr'] = learning_rate


class GCN(torch.nn.Module):
    def __init__(self, conv_1_out, conv_2_out, conv_3_out):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = ResGatedGraphConv(dataset.num_node_features, conv_1_out)
        # self.conv2 = FiLMConv(conv_1_out, conv_2_out)
        # self.conv2_1 = FiLMConv(conv_2_out, conv_2_out)
        self.conv3 = ResGatedGraphConv(conv_2_out, conv_3_out)
        self.lin = Linear(conv_3_out, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv2_1(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(8, 16, 32)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4, amsgrad=True)
criterion = torch.nn.CrossEntropyLoss()

acc_stats = pd.DataFrame(columns=['Epoch', 'Train accuracy', 'Test accuracy'])
loss_stats = pd.DataFrame(columns=['Epoch', 'Loss', 'Time'])


def save_net_state(state, filename):
    torch.save(state, filename)


def train(epoch):
    global loss_stats
    model.train()
    start_time = time.perf_counter()
    for data in train_loader:  # Iterate in batches over the training dataset.
         data = data.to(device)
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
    end_time = time.perf_counter()
    print("Loos: {}".format(loss))
    n_loss_row = {'Epoch': epoch, 'Loss': loss, 'Time': (end_time - start_time)}
    loss_stats = loss_stats.append(n_loss_row, ignore_index=True)
    loss_stats.to_csv('results/loss_net9_4.csv')


def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(device)
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def train_optim(curr_res):
    global max_result, test_failed, num_before_imp
    if curr_res > max_result:
        state_to_save = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_net_state(state_to_save, "models/test9_4.pth.tar")
        max_result = curr_res
        num_before_imp = 0
    elif num_before_imp < 10:
        num_before_imp += 1
    else:
        load_net_state("models/test9_4.pth.tar")
        print("Loading net")
        test_failed += 1
    if test_failed > 9:
        print("Finished rep times")
        sys.exit()


def train_net():
    global acc_stats
    check_every = 5
    for epoch in range(1, 1001):
        train(epoch)
        optimize_lr()
        if epoch % check_every == 0:
            train_acc = test(train_loader)
            test_acc = test(test_loader)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, lr = {learning_rate:.4f}')
            n_acc_row = {'Epoch': epoch, 'Train accuracy': train_acc, 'Test accuracy': test_acc}
            acc_stats = acc_stats.append(n_acc_row, ignore_index=True)
            acc_stats.to_csv('results/stats_net9_4.csv')
            train_optim(test_acc)


def load_net_state(state):
    print("Loading network")
    state = torch.load(state)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


def load_net_to_valid():
    load_net_state("models/test9_2.pth.tar")
    print("Train acc: {}  ".format(test(train_loader)))
    print("Test acc: {}  ".format(test(test_loader)))


# load_net_to_valid()
train_net()

# os.system("shutdown /s /t 1")