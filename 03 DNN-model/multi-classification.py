import time
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
import copy
import math

from scipy.special import softmax
import scipy.stats as ss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, recall_score, precision_score, \
    precision_recall_curve, f1_score, auc
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.model_selection import KFold, StratifiedKFold
# sns.set_theme(color_codes=True)
import warnings

warnings.filterwarnings("ignore")

random_seed = 0

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

torch.cuda.set_device("cuda:0")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# import data
mrna = pd.read_csv('D:/data/mrna.txt', sep='\t')
mir = pd.read_csv('D:/data/mirna.txt', sep='\t')
clincal = pd.read_csv('D:/data/label.txt', sep='\t')

mrna.index = mrna['gene_name']
del mrna['gene_name']

mir.index = mir['gene_name']
del mir['gene_name']

clincal.index = clincal['id']
del clincal['id']

label = clincal['label']

# select  genes with higher weight from FGL-SCCA
mrna_feature_num = 400
mir_feature_num = 200
mrna = mrna.iloc[:, :mrna_feature_num]
mir = mir.iloc[:, :mir_feature_num]
mrna = mrna.iloc[:, :mrna_feature_num]
mir = mir.iloc[:, :mir_feature_num]

mrna = mrna.values
mir = mir.values

min_max_scaler = preprocessing.MinMaxScaler()
mrna = min_max_scaler.fit_transform(mrna)
mir = min_max_scaler.fit_transform(mir)
label = label.values

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0001, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.

            verbose (bool): If True, prints a message for each validation loss improvement.

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.

            path (str): Path for the checkpoint to be saved to.

            trace_func (function): trace print function.

        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter > self.patience:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# module attention and dnn model structure
class mtlAttention(nn.Module):
    def __init__(self, In_Nodes1, In_Nodes2, Modules):
        super(mtlAttention, self).__init__()
        self.Modules = Modules
        self.sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax()

        self.task1_FC1_x = nn.Linear(In_Nodes1, Modules, bias=False)
        self.task1_FC1_y = nn.Linear(In_Nodes1, Modules, bias=False)

        self.task2_FC1_x = nn.Linear(In_Nodes2, Modules, bias=False)
        self.task2_FC1_y = nn.Linear(In_Nodes2, Modules, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.task1_FC2 = nn.Sequential(nn.Linear(Modules * 2, 32), nn.ReLU())
        self.task2_FC2 = nn.Sequential(nn.Linear(Modules * 2, 32), nn.ReLU())

        self.task1_FC3 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.task2_FC3 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())

        self.task1_FC4 = nn.Sequential(nn.Linear(16, 4), nn.Softmax())
        self.task2_FC4 = nn.Sequential(nn.Linear(16, 4), nn.Softmax())

    def forward_one(self, xg, xm):
        xg_x = self.task1_FC1_x(xg)
        xm_x = self.task2_FC1_x(xm)
        xg_y = self.task1_FC1_y(xg)
        xm_y = self.task2_FC1_y(xm)

        xg = torch.cat([xg_x.reshape(-1, 1, self.Modules), xg_y.reshape(-1, 1, self.Modules)], dim=1)
        xm = torch.cat([xm_x.reshape(-1, 1, self.Modules), xm_y.reshape(-1, 1, self.Modules)], dim=1)

        norm = torch.norm(xg, dim=1, keepdim=True)
        xg = xg.div(norm)

        norm = torch.norm(xm, dim=1, keepdim=True)
        xm = xm.div(norm)

        energy = torch.bmm(xg.reshape(-1, 2, self.Modules).permute(0, 2, 1), xm.reshape(-1, 2, self.Modules))
        attention1 = self.softmax(energy.permute(0, 2, 1)).permute(0, 2, 1)
        attention2 = self.softmax(energy).permute(0, 2, 1)

        xg_value = torch.bmm(xg, attention1)
        xm_value = torch.bmm(xm, attention2)

        xg = xg_value.view(-1, self.Modules * 2)
        xm = xm_value.view(-1, self.Modules * 2)

        xg = self.task1_FC2(xg)
        xm = self.task2_FC2(xm)
        xg = self.task1_FC3(xg)
        xm = self.task2_FC3(xm)
        xg = self.task1_FC4(xg)
        xm = self.task2_FC4(xm)

        return xg, xm


n_SKFold = KFold(n_splits=5, shuffle=True,random_state = 488)

j = 0
for train_index, test_index in n_SKFold.split(mrna):
    Xg_train, Xg_test = mrna[train_index, :], mrna[test_index, :]
    Xm_train, Xm_test = mir[train_index, :], mir[test_index, :]
    yg_train, yg_test = label[train_index], label[test_index]
    j = j + 1
    if j == 1:
        break

Xg_train = Xg_train.astype(float)
Xg_test = Xg_test.astype(float)
Xm_train = Xm_train.astype(float)
Xm_test = Xm_test.astype(float)


def cal_label(test):
    array = []
    for i in test:
        temp = np.max(i)
        count = 0
        for j in i:
            if temp == j:
                break
            count += 1
        array.append(count)
    return np.array(array)


train_losses, test_losses = [], []
start = time.time()


earlyStoppingPatience = 1000
learningRate = 0.001
weightDecay = 0.0001
num_epochs = 300000

y_train = np.array(yg_train).flatten().astype(int)
y_test = np.array(yg_test).flatten().astype(int)

Xg = torch.tensor(Xg_train, dtype=torch.float32).cuda()
Xm = torch.tensor(Xm_train, dtype=torch.float32).cuda()

Xg_test = torch.tensor(Xg_test, dtype=torch.float32).cuda()
Xm_test = torch.tensor(Xm_test, dtype=torch.float32).cuda()

y = torch.tensor(y_train, dtype=torch.float32).cuda()

ds = TensorDataset(Xg, Xm, y)
loader = DataLoader(ds, batch_size=y_train.shape[0], shuffle=True)

Xg_test = torch.tensor(Xg_test, dtype=torch.float32).cuda()
Xm_test = torch.tensor(Xm_test, dtype=torch.float32).cuda()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
In_Nodes1 = Xg_train.shape[1]
In_Nodes2 = Xm_train.shape[1]

# mtlAttention(In_Nodes1,In_Nodes2, # of module)
net = mtlAttention(In_Nodes1, In_Nodes2, 128)
net = net.to(device)
early_stopping = EarlyStopping(patience=earlyStoppingPatience, verbose=False)
optimizer = optim.Adam(net.parameters(), lr=learningRate, weight_decay=weightDecay)
loss_fn = nn.CrossEntropyLoss()

for epoch in (range(num_epochs)):
    running_loss1 = 0.0
    running_loss2 = 0.0
    for i, data in enumerate(loader, 0):
        xg, xm, y = data
        output1, output2 = net.forward_one(xg, xm)
        output1 = output1.squeeze()
        output2 = output2.squeeze()
        net.train()
        optimizer.zero_grad()
        y = y.to(torch.int64)
        loss = loss_fn(output1, y) + loss_fn(output2, y)

        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss1 += loss_fn(output1, y.view(-1)).item()
        running_loss2 += loss_fn(output2, y.view(-1)).item()

    early_stopping(running_loss1 + running_loss2, net)
    if early_stopping.early_stop:
        print("Early stopping")
        print("--------------------------------------------------------------------------------------------------")
        break

    if (epoch + 1) % 2000 == 0 or epoch == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Cross_entry_loss_task1; {:.4f}, Cross_entry_loss_task2; {:.4f}'.format(
                epoch + 1, num_epochs,
                running_loss1 + running_loss2,
                running_loss1,
                running_loss2))

### Test

test1, test2 = net.forward_one(Xg_test.clone().detach(), Xm_test.clone().detach())
test1 = test1.cpu().detach().numpy()
test2 = test2.cpu().detach().numpy()


print("=====================================result===================================")
print("pre_label:", cal_label((test1 + test2) / 2))
print("ture_label:", y_test)
print("------------------------------------")
print("ACC:", accuracy_score(y_test, cal_label((test1 + test2) / 2)))
print("F1:", f1_score(y_test, cal_label((test1 + test2) / 2), average='macro'))

print("time :", time.time() - start)
print("==================================================exit========================")
