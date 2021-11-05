
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import PyTorchToolkit as pytk
from sklearn.preprocessing import StandardScaler
import random
from torchsummary import summary
import sys
from torchviz import make_dot
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interpolate
from itertools import cycle
from sklearn.preprocessing import label_binarize

# Hyperparams
start_lr = .0008
learning_rate = .0008
min_limit = .0000001 #.000012
batch_size = 120
num_epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_vine_df = pd.read_csv('Data/DSMUM3_DATA/wine_quality/wine_train.csv')
train_vine_data = train_vine_df.copy(deep=True)
train_vine_data.drop(['label'], axis=1, inplace=True)

sk_scaler = StandardScaler().fit(train_vine_data)
train_vine_data = sk_scaler.transform(train_vine_data)

train_tensor_data = torch.from_numpy(train_vine_data).type(torch.FloatTensor).to(device).cuda()  #  skylear ncross entropy
train_tensor_labels = torch.from_numpy(train_vine_df['label'].values).type(torch.LongTensor).to(device).cuda()
train_dataset = TensorDataset(train_tensor_data, train_tensor_labels)
test_vine_df = pd.read_csv('Data/DSMUM3_DATA/wine_quality/wine_test.csv')
test_vine_data = test_vine_df.copy(deep=True)
test_vine_data.drop(['label'], axis=1, inplace=True)
sk_scaler = StandardScaler().fit(test_vine_data)
test_vine_data = sk_scaler.transform(test_vine_data)

test_tensor_data = torch.from_numpy(test_vine_data).type(torch.FloatTensor).to(device).cuda()
test_tensor_labels = torch.from_numpy(test_vine_df['label'].values).type(torch.LongTensor).to(device).cuda()
# print("Labels: {}".format(test_tensor_labels))
test_dataset = TensorDataset(test_tensor_data, test_tensor_labels)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class TestCCN2(nn.Module):
    def __init__(self, num_in, cnn1_out, cnn2_out, fc1_out, drop_1, drop_2):
        super(TestCCN2, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=num_in, out_channels=cnn1_out, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=cnn1_out, out_channels=cnn2_out, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(cnn2_out, fc1_out)
        self.fc2 = nn.Linear(fc1_out, 6)

        self.batchnorm1d_1 = nn.BatchNorm1d(cnn1_out)
        self.batchnorm1d_2 = nn.BatchNorm1d(cnn2_out)
        self.batchnorm1d_3 = nn.BatchNorm1d(fc1_out)
        self.dropout_1 = nn.Dropout(drop_1)
        self.dropout_2 = nn.Dropout(drop_2)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = torch.tanh(self.conv1d_1(x))
        x = self.batchnorm1d_1(x)
        x = torch.tanh(self.conv1d_2(x))
        x = self.batchnorm1d_2(x)
        x = x.flatten(1)
        x = self.batchnorm1d_3(torch.tanh(self.fc1(x)))
        x = self.dropout_1(x)
        x = self.fc2(x)
        return x

    def init_layers(self):
        nn.init.xavier_uniform(self.conv1d_1.weight)
        nn.init.constant(self.conv1d_1.bias, .001)
        nn.init.xavier_uniform(self.conv1d_2.weight)
        nn.init.constant(self.conv1d_2.bias, .001)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.constant(self.fc1.bias, .001)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.constant(self.fc2.bias, .001)


class TestFCN10(nn.Module):
    def __init__(self, nuym_in, fc1_out, fc2_out, drop_1, drop_2):
        super(TestFCN10, self).__init__()
        self.fc1 = nn.Linear(nuym_in, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 6)
        self.batchnorm1d_1 = nn.BatchNorm1d(fc1_out)
        self.batchnorm1d_2 = nn.BatchNorm1d(fc2_out)
        self.dropout_1 = nn.Dropout(p=drop_1)
        self.dropout_2 = nn.Dropout(p=drop_2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.batchnorm1d_1(x)
        x = self.dropout_1(x)
        x = torch.tanh(self.fc2(x))
        x = self.batchnorm1d_2(x)
        x = self.dropout_2(x)
        x = self.fc3(x)
        return x

    def init_layers(self):
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.constant(self.fc1.bias, .0001)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.constant(self.fc2.bias, .0001)
        nn.init.xavier_uniform(self.fc3.weight)
        nn.init.constant(self.fc3.bias, .0001)


my_model = TestFCN10(11, 84, 164, 0.2, 0.2)
# my_model = TestCCN2(11, 16, 16, 32, 0.25, 0.2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(my_model.parameters(), lr=learning_rate, weight_decay=0.001, amsgrad=True)
my_model.to(device)
# summary(my_model, (1, 11))
# sys.exit()


def save_net_state(state, filename):
    torch.save(state, filename)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def calculate_lr(curr_lr, past_result, curr_result):
    global min_limit
    step_1 = .999
    step_2 = .996
    step_3 = .997
    if curr_lr > min_limit:
        if curr_result/past_result < .8:
            return curr_lr
        elif curr_result/past_result < .9 and curr_lr * step_1 > min_limit:
            return curr_lr * step_1
        elif curr_result/past_result < 1 and curr_lr * step_2 > min_limit:
            return curr_lr * step_2
        elif curr_lr * step_3 > min_limit:
            return curr_lr * step_3
        else:
            return curr_lr
    else:
        return curr_lr


def optimize_lr(past_result, curr_result):
    global learning_rate
    learning_rate = calculate_lr(learning_rate, past_result, curr_result)
    for g in optimizer.param_groups:
        g['lr'] = learning_rate


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


loss_stats = pd.DataFrame(columns=['Loss on Epoch'])

max_stat = 0


def train_model():
    global loss_stats
    global learning_rate
    net_train_stat = pd.DataFrame(columns=['loss', 'train_acc', 'test_acc'])
    checkpoint_num = 250
    loss_count = 0
    iter_count = 0
    for epoch in range(num_epochs):
        for batch_index, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = my_model(data)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_count += loss.item()
            iter_count += 1
        print("Epoch num: {} Loss is: {} Curr LR: {}".format(epoch, loss_count/iter_count, learning_rate))
        new_row = {'Loss on epoch': (loss_count/iter_count)}
        loss_stats = loss_stats.append(new_row, ignore_index=True)
        if epoch > 0:
            optimize_lr(loss_stats.iloc[epoch-1]['Loss on epoch'], loss_stats.iloc[epoch]['Loss on epoch'])

            # if epoch % checkpoint_num == 0:
            #     curr_loc = int(epoch/checkpoint_num)
            #     new_row = {'loos': (loss_count / iter_count), 'train_acc': check_accuracy(train_loader, my_model), 'test_acc': check_accuracy(test_loader, my_model)}
            #     net_train_stat = net_train_stat.append(new_row, ignore_index=True)
            #     if curr_loc > 1:
            #         if max(net_train_stat['test_acc']) == net_train_stat.iloc[curr_loc-1]['test_acc']:
            #             print(max(net_train_stat['test_acc']), "ekalsss: ", net_train_stat.iloc[curr_loc-1]['test_acc'])
            #             state_to_save = {'state_dict': my_model.state_dict(), 'optimizer': optimizer.state_dict()}
            #             save_net_state(state_to_save, "Data/Networks/FCNN_zad5_1.pth.tar")
            #             if max(net_train_stat['test_acc']) > max_stat:
            #                 net_train_stat.to_csv('Data/Results/FCNN_zad5_1_net_train_stats')
            #         else:
            #             load_net_state("Data/Networks/FCNN_zad5_1.pth.tar")
            #             print("Sth is wong")
        loss_count = 0
        iter_count = 0
    loss_stats.to_csv('Data/Results/FCNN_zad5_1_net_loss_stats.csv')

    check_accuracy(train_loader, my_model)
    check_accuracy(test_loader, my_model)


def load_net_state(state):
    print("Loading network")
    state = torch.load(state)
    my_model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


def load_net_to_valid():
    load_net_state("Data/Networks/FCNN_10_1_B4.pth.tar")
    check_accuracy(train_loader, my_model)
    check_accuracy(test_loader, my_model)
    create_confusion_matrix(train_loader, my_model)
    create_confusion_matrix(test_loader, my_model)


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
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def print_stats(y_true, y_pred, confusion):
    r1 = metrics.recall_score(y_true, y_pred, average="macro")
    p1 = metrics.precision_score(y_true, y_pred, average="macro")
    #print(metrics.classification_report(y_true, y_pred, target_names=['0', '1', '2', '3', '4', '5']))
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    s1 = TN/(TN+FP)
    #print("FP {}, FN {}, TP {}, TN {}".format(FP, FN, TP, TN))
    print("Sensitivity: {}  Precision: {} Specifity {}".format(r1, p1, np.mean(s1)))

    y_true = label_binarize(y_true, classes=[0,1,2,3,4,5])
    y_pred = label_binarize(y_pred, classes=[0,1,2,3,4,5])

    print(y_true)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(6):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(6):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 6

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(6), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


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
        print_stats(results.cpu(), predictions.cpu(), cm)
        classes = list(range(6))
        plt.figure(figsize=(6, 6))
        plot_confusion_matrix(cm, classes)
        plt.show()


net_blueprint = pd.read_csv('Data/DSMUM3_DATA/wine_quality/net_blueprint_4.csv')
net_stats = pd.DataFrame(columns=['Training_Serie', 'train_acc', 'test_acc'])


def val_n_times(n):
    global start_lr
    global learning_rate
    global min_limit
    global net_stats
    global my_model
    global criterion
    global optimizer
    global max_stat
    for i in range(n):
        print("TRAIN ITER NUM: {}".format(i))
        start_lr = net_blueprint['start_lr'][i]
        learning_rate = net_blueprint['lr'][i]
        min_limit = net_blueprint['min_limit'][i]
        my_model = TestFCN10(11, net_blueprint['fc1_out'][i], net_blueprint['fc2_out'][i], net_blueprint['drop_1'][i], net_blueprint['drop_2'][i])
        #my_model = TestFCN10()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(my_model.parameters(), lr=learning_rate, weight_decay=0.01, amsgrad=True)
        my_model.to(device)
        my_model.apply(weight_reset)
        my_model.init_layers()
        my_model.train()
        train_model()
        new_stat_row = {'Training_Serie': i, 'train_acc': check_accuracy(train_loader, my_model), 'test_acc': check_accuracy(test_loader, my_model)}
        net_stats = net_stats.append(new_stat_row, ignore_index=True)
        net_stats.to_csv('Data/Results/net_stats_zad5_test.csv')
        loss_stats.to_csv('Data/Results/loos_stats_zad5_test.csv')
        if i > 0 and net_stats['test_acc'][i] > max_stat:
            max_stat = net_stats['test_acc'][i]
            state_to_save = {'state_dict': my_model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_net_state(state_to_save, "Data/Networks/FCNN_zad5_one.pth.tar")


#val_n_times(len(net_blueprint['lr']))
# val_n_times(1)
#train_model()
load_net_to_valid()