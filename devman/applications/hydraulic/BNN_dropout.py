# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import pyro.poutine as poutine

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class NN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.out(output)
        return output

#  A pytorch generic function that takes a data.Dataset object and splits it to validation and training efficiently.


class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    '''
    if random_seed != None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    valid_size = dslen // split_fold
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)

    return train, valid

#   Training Dataset


class MyDataset1(Dataset):

    def __init__(self):
        array = np.loadtxt('Accum_Train')
        # array = np.loadtxt('Accum_TrainPr20Me')
        col = np.shape(array)[1] - 1
        # training data
        X = array[:, 0: col]
        Y = array[:, col]

        self.len = array.shape[0]

        self.x_data = torch.from_numpy(X).float()
        # 第一列为标签为，存在y_data中
        self.y_data = torch.from_numpy(Y).long()

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        # print(self.x_data[index])
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len

# testing data


class MyDataset2(Dataset):

    def __init__(self):
        array = np.loadtxt('Accum_Test')
        # array = np.loadtxt('Accum_TestPr20Me')
        col = np.shape(array)[1] - 1
        # training data
        X = array[:, 0: col]
        Y = array[:, col]

        self.len = array.shape[0]

        self.x_data = torch.from_numpy(X).float()
        # 第一列为标签为，存在y_data中
        self.y_data = torch.from_numpy(Y).long()

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        # print(self.x_data[index])
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


dataset1 = MyDataset1()  # training data set without validation split
dataset2 = MyDataset2()  # testing set

# 20个维度的输入，隐藏层50个神经元，3个维度的输出
net = NN(20, 100, 4)

log_softmax = nn.LogSoftmax(dim=1)


def model(x_data, y_data):
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight),
                        scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias),
                        scale=torch.ones_like(net.fc1.bias))

    outw_prior = Normal(loc=torch.zeros_like(net.out.weight),
                        scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias),
                        scale=torch.ones_like(net.out.bias))

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
              'out.weight': outw_prior, 'out.bias': outb_prior}

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = log_softmax(lifted_reg_model(x_data))

    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)


# It helps us initialize a well-behaved distribution that
# later we can optimize to approximate the true posterior.
softplus = torch.nn.Softplus()


def mc_elbo_with_l2(model, guide, data1, data2, lam=0.01):
    guide_trace = poutine.trace(guide).get_trace(data1, data2)
    model_trace = poutine.trace(
        poutine.replay(model, trace=guide_trace)).get_trace(data1, data2)
    logp = model_trace.log_prob_sum()
    logq = guide_trace.log_prob_sum()
    penalty = 0
# 加入正则化
#     for node in model_trace.nodes.values():
#         if node["type"] == "param":
#             penalty = penalty + lam * torch.sum(torch.pow(node["value"], 2))
#
#     for node in guide_trace.nodes.values():
#         if node["type"] == "param":
#             penalty = penalty + lam * torch.sum(torch.pow(node["value"], 2))
#
    return logq - logp + penalty


def guide(x_data, y_data):
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param,
                        scale=outw_sigma_param).independent(1)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
              'out.weight': outw_prior, 'out.bias': outb_prior}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()


optim = Adam({"lr": 0.01})  # learning rate 0.01
# svi = SVI(model, guide, optim, loss=Trace_ELBO())
svi = SVI(model, guide, optim, loss=mc_elbo_with_l2)

num_samples = 10


def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)


datasetTrain, datasetVal = train_valid_split(
    dataset1)  # training set & validation set
train_loader = DataLoader(dataset=datasetTrain,
                          batch_size=100, shuffle=True)  # 训练集
# 验证集 for hyperparameter tuning
validation_loader = DataLoader(dataset=datasetVal, batch_size=20, shuffle=True)
test_loader = DataLoader(dataset=dataset2)


num_iterations = 250  # 需要调整 stopping criterion
loss = 0
loss_train_epoch = np.zeros(num_iterations)
loss_valid_epoch = np.zeros(num_iterations)
lam = 0

for j in range(num_iterations):
    loss_train = 0
    for batch_id, data in enumerate(train_loader):
         # calculate the loss and take a gradient step
        loss_train += svi.step(data[0], data[1], lam)  #
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss_train / normalizer_train
    loss_train_epoch[j] = total_epoch_loss_train
    print("Epoch ", j, " Loss ", total_epoch_loss_train)

    loss_valid = 0
    for mini_id, data in enumerate(validation_loader):
        loss_valid += svi.evaluate_loss(data[0], data[1], lam)
    normalizer_valid = len(validation_loader.dataset)
    loss_valid_epoch[j] = loss_valid / normalizer_valid

np.savetxt('BNN_dp_trainloss', loss_train_epoch)
np.savetxt('BNN_dp_validloss', loss_valid_epoch)

# Plot training and tesiting loss
sns.set_style("whitegrid")
plt.figure(figsize=(14, 4))
plt.title('Loss on Training and Validation Set')
plt.plot(np.arange(0, num_iterations), loss_train_epoch, label='Training Loss')
plt.plot(np.arange(0, num_iterations),
         loss_valid_epoch, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('Prediction when network is forced to predict on validation set')
correct = 0
total = 0
# arr = np.loadtxt('Accum_Test')
arr = np.loadtxt('Accum_TestPr20Me')

Y_pred = np.zeros(np.shape(arr)[0])
Y_test = arr[:, np.shape(arr)[1] - 1]


for j, data in enumerate(test_loader):
    status, labels = data
    predicted = predict(status)
    total += labels.size(0)
    correct += (predicted == labels.numpy()).sum().item()
    Y_pred[j] = np.array(predicted)
print("accuracy: %.3f %%" % (100 * correct / total))
