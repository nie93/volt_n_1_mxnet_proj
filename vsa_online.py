
# coding: utf-8

# In[1]:


# https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html#Faster-modeling-with-gluon.nn.Sequential
get_ipython().magic(u'matplotlib inline')

from __future__ import print_function
from time import time
from IPython.display import clear_output
import csv
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd, autograd, gluon

X0 = []
y0 = []
with open('snapshots_Xy_180810_114422.csv', 'rb') as f:
    rdr = csv.reader(f, delimiter=',')
    next(rdr)
    for row in rdr:
        X0.append([float(x) for x in row[1:-1]])
        y0.append(int(row[1]))


X1 = []
y1 = []
with open('snapshots_Xy_180723_141339.csv', 'rb') as f:
    rdr = csv.reader(f, delimiter=',')
    next(rdr)
    for row in rdr:
        X1.append([float(x) for x in row[1:-1]])
        y1.append(int(row[1]))
    
        
X0 = nd.array(X0)
y0 = nd.array(y0)      
X1 = nd.array(X1)
y1 = nd.array(y1)

mx.random.seed(4222)

sel = nd.random.shuffle(nd.arange(X0.shape[0]))
X0 = X0[sel,:]
y0 = y0[sel]

sel = nd.random.shuffle(nd.arange(X1.shape[0]))
X1 = X1[sel,:]
y1 = y1[sel]




num_of_samples, num_of_feats = X0.shape
num_of_samples = int(0.7 * num_of_samples)
batch_size = int(num_of_samples / 1)

X0train = X0[0:num_of_samples,:]
y0train = y0[0:num_of_samples]
X0test = X0[num_of_samples:,:]
y0test = y0[num_of_samples:]

print(' NUMBER OF FEATURES - %4d' %num_of_feats)
print('TRAINING DATASET #0 - SECURE: %4d | INSECURE: %4d' 
      %(sum(y0train==1).asscalar(), sum(y0train==0).asscalar()))
print(' TESTING DATASET #0 - SECURE: %4d | INSECURE: %4d' 
      %(sum(y0test==1).asscalar(), sum(y0test==0).asscalar()))

X1train = X1[0:num_of_samples,:]
y1train = y1[0:num_of_samples]
X1test = X1[num_of_samples:,:]
y1test = y1[num_of_samples:]


print(' NUMBER OF FEATURES - %4d' %num_of_feats)
print('TRAINING DATASET #1 - SECURE: %4d | INSECURE: %4d' 
      %(sum(y1train==1).asscalar(), sum(y1train==0).asscalar()))
print(' TESTING DATASET #1 - SECURE: %4d | INSECURE: %4d' 
      %(sum(y1test==1).asscalar(), sum(y1test==0).asscalar()))


train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X0train, y0train),
                                   batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X0test, y0test),
                                  batch_size=batch_size, shuffle=True)


update_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X1train, y1train),
                                   batch_size=batch_size, shuffle=True)
update_test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X1test, y1test),
                                  batch_size=batch_size, shuffle=True)

mdl_ctx = mx.gpu()
dat_ctx = mx.gpu()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(mdl_ctx).reshape((-1, num_of_feats))
        label = label.as_in_context(mdl_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

def movavg(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec


# In[2]:


def nntrain2(algo, e1, e2, param):
    train_acc = []
    test_acc = []
    epo_time = []
    epo_loss = []
    cum_time = 0.
    
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(256, activation="relu"))
        net.add(gluon.nn.Dense(128, activation="relu"))
        net.add(gluon.nn.Dense(64))

    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=mdl_ctx)
    trainer = gluon.Trainer(net.collect_params(), algo, param)

    for e in range(e1):
        tic = time()
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(mdl_ctx).reshape((-1, num_of_feats))
            label = label.as_in_context(mdl_ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()

        train_acc.append(evaluate_accuracy(train_data, net))
        test_acc.append(evaluate_accuracy(test_data, net))
        epo_time.append(time() - tic)
        cum_time += time() - tic
        epo_loss.append(cumulative_loss/num_of_samples)
        
        clear_output(wait=True)
        print("Epoch #%4d. Train_acc: %9.8f, Test_acc: %9.8f, Time: %9.8f, Loss: %12.8f" %
              (e+1, train_acc[e], test_acc[e], epo_time[e], epo_loss[e]))                

    for e in range(e2):
        tic = time()
        cumulative_loss = 0
        for i, (data, label) in enumerate(update_data):
            data = data.as_in_context(mdl_ctx).reshape((-1, num_of_feats))
            label = label.as_in_context(mdl_ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()

        train_acc.append(evaluate_accuracy(update_data, net))
        test_acc.append(evaluate_accuracy(update_test_data, net))
        epo_time.append(time() - tic)
        cum_time += time() - tic
        epo_loss.append(cumulative_loss/num_of_samples)
        
        clear_output(wait=True)
        print("Epoch #%4d. Train_acc: %9.8f, Test_acc: %9.8f, Time: %9.8f, Loss: %12.8f" %
              (e+1, train_acc[e], test_acc[e], epo_time[e], epo_loss[e]))    
    
    print("    >>> Completed! Training time: %9.8f" %cum_time)
    for i in range((e1+e2)/500):
        idx = 500 * (i+1)
        print('With %4d Epochs: train_acc: %9.8f, test_acc: %9.8f, loss: %9.8f, cum_time: %8.4fs' 
              %(idx, train_acc[idx-1], test_acc[idx-1], epo_loss[idx-1], sum(epo_time[0:idx-1])))
    
    print('Preset Learning Rate: %s' %trainer._optimizer._get_lr(-1))
    return train_acc, test_acc, epo_time, epo_loss, cum_time    


# In[3]:


ma = 50
num_of_train_epochs = ma + 2000
num_of_update_epochs = ma + 4000
nonadarate = 2e-5
adarate = 2e-4


# In[4]:


# **************** SGD ****************
sgd_train_acc, sgd_test_acc, sgd_epo_time, sgd_epo_loss, sgd_cum_time = nntrain2('sgd', num_of_train_epochs,
                                                                                 num_of_update_epochs,
                                                                                {'learning_rate': nonadarate})


# In[5]:


# **************** SGD-M ****************
sgdm_train_acc, sgdm_test_acc, sgdm_epo_time, sgdm_epo_loss, sgdm_cum_time = nntrain2('sgd', num_of_train_epochs,
                                                                                      num_of_update_epochs,
                                                                                      {'learning_rate': nonadarate,
                                                                                       'momentum': 0.009})


# In[6]:


# **************** NAG ****************
nag_train_acc, nag_test_acc, nag_epo_time, nag_epo_loss, nag_cum_time = nntrain2('nag', num_of_train_epochs,
                                                                                 num_of_update_epochs, 
                                                                                 {'learning_rate': nonadarate,
                                                                                  'momentum': 0.})


# In[7]:


# **************** NAG-M ****************
nagm_train_acc, nagm_test_acc, nagm_epo_time, nagm_epo_loss, nagm_cum_time = nntrain2('nag', num_of_train_epochs,
                                                                                      num_of_update_epochs,
                                                                                      {'learning_rate': nonadarate,
                                                                                       'momentum': 0.009})


# In[8]:


# # **************** TestAlgo ****************
# algo_train_acc, algo_test_acc, algo_epo_time, algo_epo_loss, algo_cum_time = nntrain2('rmsprop', num_of_train_epochs,
#                                                                                       num_of_update_epochs,
#                                                                                       {'learning_rate': nonadarate})


# In[9]:


# **************** AdaGrad ****************
adag_train_acc, adag_test_acc, adag_epo_time, adag_epo_loss, adag_cum_time = nntrain2('adagrad', num_of_train_epochs,
                                                                                      num_of_update_epochs,
                                                                                      {'learning_rate': adarate})


# In[10]:


# **************** ADAM ****************
adam_train_acc, adam_test_acc, adam_epo_time, adam_epo_loss, adam_cum_time = nntrain2('adam', num_of_train_epochs,
                                                                                      num_of_update_epochs, 
                                                                                      {'learning_rate': adarate})


# In[11]:


# **************** NADAM ****************
nadam_train_acc, nadam_test_acc, nadam_epo_time, nadam_epo_loss, nadam_cum_time = nntrain2('nadam', num_of_train_epochs,
                                                                                           num_of_update_epochs, 
                                                                                           {'learning_rate': adarate})


# In[33]:


# **************** CombinedPlots ****************
fig_combined, ((ax11,ax12), 
               (ax21,ax22), 
               (ax31,ax32)) = plt.subplots(3, 2, sharex='col', figsize=(13, 7))

ax11.grid(True)
ax11.plot(movavg(sgd_train_acc, ma))
ax11.plot(movavg(nag_train_acc, ma))
ax11.plot(movavg(sgdm_train_acc, ma), ls='--')
ax11.plot(movavg(nagm_train_acc, ma), ls='--')
ax11.legend(['SGD', 'NAG', 'SGD-m', 'NAG-m'], loc='lower right')
ax11.set_title('(a) Non-adaptive Learning Algorithms')
ax11.set_ylabel('Training Accuracy')
ax11.set_ylim([0.64, 1.01])
# ax11.set_xlim([-100, 2100])

ax12.grid(True)
ax12.plot(movavg(adag_train_acc, ma))
ax12.plot(movavg(adam_train_acc, ma))
ax12.plot(movavg(nadam_train_acc, ma))
# ax12.plot(movavg(algo_train_acc, ma))
ax12.legend(['AdaGrad', 'Adam', 'Nadam'], loc='lower right')
ax12.set_title('(b) Adaptive Learning Algorithms')
ax12.set_ylim([0.64, 1.01])

ax21.grid(True)
ax21.plot(movavg(sgd_test_acc, ma))
ax21.plot(movavg(nag_test_acc, ma))
ax21.plot(movavg(sgdm_test_acc, ma), ls='--')
ax21.plot(movavg(nagm_test_acc, ma), ls='--')
ax21.legend(['SGD', 'NAG', 'SGD-m', 'NAG-m'], loc='lower right')
ax21.set_ylabel('Testing Accuracy')
ax21.set_ylim([0.64, 1.01])
# ax21.set_xlim([-100, 2100])

ax22.grid(True)
ax22.plot(movavg(adag_test_acc, ma))
ax22.plot(movavg(adam_test_acc, ma))
ax22.plot(movavg(nadam_test_acc, ma))
# ax22.plot(movavg(algo_test_acc, ma))
ax22.legend(['AdaGrad', 'Adam', 'Nadam'], loc='lower right')
ax22.set_ylim([0.64, 1.01])

ax31.grid(True)
ax31.set_xlabel('Number of Epochs')
ax31.set_ylabel('Entropy Loss')
ax31.plot(movavg(sgd_epo_loss,ma))
ax31.plot(movavg(nag_epo_loss,ma))
ax31.plot(movavg(sgdm_epo_loss, ma), ls='--')
ax31.plot(movavg(nagm_epo_loss, ma), ls='--')
ax31.set_yscale("log", nonposy='clip')
ax31.legend(['SGD', 'NAG', 'SGD-m', 'NAG-m'], loc='upper right')

ax32.grid(True)
ax32.set_xlabel('Number of Epochs')
ax32.plot(movavg(adag_epo_loss,ma))
ax32.plot(movavg(adam_epo_loss,ma))
ax32.plot(movavg(nadam_epo_loss,ma))
ax32.set_yscale("log", nonposy='clip')
ax32.legend(['AdaGrad', 'Adam', 'Nadam'], loc='upper right')



ax11.axvline(x=2000, c='tab:purple', ls='--')
ax21.axvline(x=2000, c='tab:purple', ls='--')
ax31.axvline(x=2000, c='tab:purple', ls='--')
ax12.axvline(x=2000, c='tab:purple', ls='--')
ax22.axvline(x=2000, c='tab:purple', ls='--')
ax32.axvline(x=2000, c='tab:purple', ls='--')


fig_combined.tight_layout()
fig_combined.savefig('combined.eps', format='eps')


# In[16]:


fig_time, (axt11,axt12) = plt.subplots(1,2, sharex='col', sharey='row', figsize=(13, 3))

axt11.grid(True)
axt11.plot(movavg(sgd_epo_time, 20*ma))
axt11.plot(movavg(sgdm_epo_time, 20*ma))
axt11.plot(movavg(nag_epo_time, 20*ma))
axt11.plot(movavg(nagm_epo_time, 20*ma))
axt11.set_xlabel('Number of Epochs')
axt11.set_ylabel('Training Time')
axt11.legend(['SGD', 'NAG', 'SGD-m', 'NAG-m'], loc='upper right')

axt12.grid(True)
axt12.plot(movavg(adag_epo_time, 20*ma))
axt12.plot(movavg(adam_epo_time, 20*ma))
axt12.plot(movavg(nadam_epo_time, 20*ma))
axt12.legend(['AdaGrad', 'Adam', 'Nadam'], loc='upper right')
axt12.set_xlabel('Number of Epochs')

fig_time.tight_layout()
fig_time.savefig('combined_time.eps', format='eps')

