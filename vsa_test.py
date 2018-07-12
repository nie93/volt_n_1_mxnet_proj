
# coding: utf-8

# In[1]:


# https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html#Faster-modeling-with-gluon.nn.Sequential
from __future__ import print_function
import csv
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon


# In[2]:


batch_size = 2000
num_of_samples = 4000


# In[3]:


X = []
y = []
with open('snapshots_Xy_180709_124937.csv', 'rb') as f:
    rdr = csv.reader(f, delimiter=',')
    next(rdr)
    # dat = [r for r in rdr]
    for row in rdr:
        X.append([float(x) for x in row[1:-1]])
        y.append(int(row[1]))

X = nd.array(X)
y = nd.array(y)

Xtrain = X[0:3000,:]
Xtest = X[3000:,:]
ytrain = y[0:3000]
ytest = y[3000:]

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtrain, ytrain),
                                   batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtest, ytest),
                                  batch_size=batch_size, shuffle=True)


# In[4]:


mdl_ctx = mx.cpu()
dat_ctx = mx.cpu()


# In[5]:


num_hidden = 80
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(256, activation="relu"))
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(64, activation="relu"))
    net.add(gluon.nn.Dense(32))

net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=mdl_ctx)
trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate': .01})


# In[6]:


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


# In[7]:


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(mdl_ctx).reshape((-1, 337))
        label = label.as_in_context(mdl_ctx).reshape((-1))
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# In[8]:


epochs = 10
smoothing_constant = .01

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(mdl_ctx).reshape((-1, 337))
        label = label.as_in_context(mdl_ctx).reshape((-1))
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()


    train_accuracy = evaluate_accuracy(train_data, net)
    test_accuracy = evaluate_accuracy(test_data, net)
    print("Epoch %2d. Train_acc: %12.8f, Test_acc: %12.8f, Loss: %s" %
          (e, train_accuracy, test_accuracy, cumulative_loss/num_of_samples))

