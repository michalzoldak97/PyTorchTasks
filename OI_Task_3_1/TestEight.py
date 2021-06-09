from torch_geometric.datasets import Planetoid, MNISTSuperpixels
import torch
import torch.nn.functional as F
from torch_geometric.nn import  FiLMConv, ResGatedGraphConv
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
from torch_geometric.nn import GraphConv

dataset = MNISTSuperpixels(root='/tmp/MNIST/test', train=False)

train_dataset = MNISTSuperpixels(root='/tmp/MNIST/train', train=True)
test_dataset = MNISTSuperpixels(root='/tmp/MNIST/test', train=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

learning_rate = .01


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
        self.conv2 = FiLMConv(conv_1_out, conv_2_out)
        self.conv3 = ResGatedGraphConv(conv_2_out, conv_3_out)
        self.lin = Linear(conv_3_out, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def save_net_state(state, filename):
    torch.save(state, filename)

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         data = data.to(device)
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
    print("Loos: {}".format(loss))


def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(device)
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(200):
    train()
    optimize_lr()
    if epoch % 5 == 0:
        state_to_save = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_net_state(state_to_save, "models/test8_1.pth.tar")
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, lr = {learning_rate:.4f}')
