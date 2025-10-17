import numpy as np
import random
import pickle
import platform
import os
import torch
torch.manual_seed(1)
np.random.seed(1)


def load_pickle(f):
    version=platform.python_version_tuple()
    if version[0]== '2':
        return pickle.load(f)
    elif version[0]== '3':
        return pickle.load(f,encoding='latin1')
    raise ValueError("invalid python version:{}".format(version))
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict=load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(ROOT,'data_batch_%d'%(b,))
        X,Y=load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X,Y
    Xte,Yte=load_CIFAR_batch(os.path.join(ROOT,'test_batch'))
    return Xtr, Ytr, Xte, Yte
def read_data_cifar_10_all():
    num_data = 50000
    P1 = 0
    P2 = 1
    P3 = 2
    P4 = 3
    P5 = 4
    P6 = 9
    P7 = 6
    P8 = 7
    P9 = 8
    A = 5
    random.seed(1)
    datasets = 'data/cifar10/cifar-10-batches-py'
    train_data, train_label, test_data, test_label = load_CIFAR10(datasets)

    train_data = np.swapaxes(train_data, 1, 3)
    train_data = np.swapaxes(train_data, 2, 3)

    test_data = np.swapaxes(test_data, 1, 3)
    test_data = np.swapaxes(test_data, 2, 3)

    train_label = torch.tensor(train_label)
    test_label = torch.tensor(test_label)
    train_data = torch.tensor(train_data)
    test_data = torch.tensor(test_data)

    rand_index = random.sample(range(train_data.size()[0]), num_data)
    new_train_label = train_label.clone()

    for i in range(train_data.size()[0]):
        if train_label[i] == P1:
            new_train_label[i] = 0
        if train_label[i] == P2:
            new_train_label[i] = 1
        if train_label[i] == P3:
            new_train_label[i] = 2
        if train_label[i] == P4:
            new_train_label[i] = 3
        if train_label[i] == P5:
            new_train_label[i] = 4
        if train_label[i] == P6:
            new_train_label[i] = 5
        if train_label[i] == P7:
            new_train_label[i] = 6
        if train_label[i] == P8:
            new_train_label[i] = 7
        if train_label[i] == P9:
            new_train_label[i] = 8
        if train_label[i] == A:
            new_train_label[i] = 9
    train_label = new_train_label.clone()
    new_train_label = train_label.clone()
    num_unlabel = torch.zeros(10)

    for i in range(train_data.size()[0]):
        if i in rand_index[:int(num_data / 2)]:
            new_train_label[i] = 9

    for i in range(10):
        a = torch.eq(new_train_label, 9).long()
        b = torch.eq(train_label, i).long()

        num_unlabel[i] = torch.sum(a & b)

    num_A = torch.eq(new_train_label, 9).sum()

    new_test_label = test_label.clone()
    for i in range(test_data.size()[0]):
        if test_label[i] == P1:
            new_test_label[i] = 0
        if test_label[i] == P2:
            new_test_label[i] = 1
        if test_label[i] == P3:
            new_test_label[i] = 2
        if test_label[i] == P4:
            new_test_label[i] = 3
        if test_label[i] == P5:
            new_test_label[i] = 4
        if test_label[i] == P6:
            new_test_label[i] = 5
        if test_label[i] == P7:
            new_test_label[i] = 6
        if test_label[i] == P8:
            new_test_label[i] = 7
        if test_label[i] == P9:
            new_test_label[i] = 8
        if test_label[i] == A:
            new_test_label[i] = 9


    data_prior = num_unlabel.float() / num_A

    print('data_prior',data_prior)


    print('train_x shape:%s, train_y shape:%s' % (train_data.shape, train_label.shape))
    print('test_x shape:%s, test_y shape:%s' % (test_data.shape, test_label.shape))
    return train_data, new_train_label, test_data, new_test_label, data_prior