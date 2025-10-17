import os
import scipy.io as sio
import random
import numpy as np
import torchvision.datasets.mnist as mnist
import torch


def read_data_SVHN(data_set):
    num_data = 5000
    random.seed(1)
    P = 7
    N = 8
    A = 9
    root="D:/code/data/SVHN/"
    train = sio.loadmat(root + "/train_32x32.mat")
    test = sio.loadmat(root + '/test_32x32.mat')

    train_data = train['X']
    train_label = train['y']
    test_data = test['X']
    test_label = test['y']

    train_data = np.swapaxes(train_data, 0, 3)
    train_data = np.swapaxes(train_data, 1, 2)
    train_data = np.swapaxes(train_data, 2, 3)

    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 1, 2)
    test_data = np.swapaxes(test_data, 2, 3)

    train_label = train_label.reshape(73257, )
    test_label = test_label.reshape(26032, )

    train_label = torch.tensor(train_label)
    test_label = torch.tensor(test_label)
    train_data = torch.tensor(train_data)
    test_data = torch.tensor(test_data)
    data_prior = [0, 0, 0, 0, 0, 0]
    num_unlabel_known = 0
    num_unknown_known = 0
    num_unknown_unlabel = 0
    num_unlabel_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == P or train_label[i] == N or train_label[i] == A:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = new_train_label.clone()
    rand_index = random.sample(range(new_train_set.size()[0]), num_data)
    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P:
            new_train_label[i] = 0
        if new_train_label_index[i] == N:
            new_train_label[i] = 1
        if new_train_label_index[i] == A:
            new_train_label[i] = 2
        # print(new_train_label[i])
        if i in rand_index[:int(num_data / 2)] and (new_train_label[i] == 0 or new_train_label[i] == 1):
            if new_train_label[i] == 0:
                num_unlabel_known = num_unlabel_known + 1
            elif new_train_label[i] == 1:
                num_unlabel_unlabel = num_unlabel_unlabel + 1
            new_train_label[i] = 1
        if i in rand_index[int(num_data / 2):]:
            if new_train_label[i] == 0:
                num_unknown_known = num_unknown_known + 1
            elif new_train_label[i] == 1:
                num_unknown_unlabel = num_unknown_unlabel + 1
            new_train_label[i] = 2
    num_known = torch.eq(new_train_label, 0).sum()
    num_unlabel = torch.eq(new_train_label, 1).sum()
    num_unknown = torch.eq(new_train_label, 2).sum()
    num_known_class = torch.eq(train_label, P).sum()
    num_unlabel_class = torch.eq(train_label, N).sum()
    num_unknown_class = torch.eq(train_label, A).sum()

    num_unlabel_known = torch.tensor(num_unlabel_known)
    num_unknown_known = torch.tensor(num_unknown_known)
    num_unknown_unlabel = torch.tensor(num_unknown_unlabel)
    num_unlabel_unlabel = torch.tensor(num_unlabel_unlabel)

    data_prior[0] = torch.tensor(1)
    data_prior[1] = num_unlabel_known.float() / num_unlabel
    data_prior[2] = (num_unlabel_class - num_unknown_unlabel).float() / num_unlabel
    data_prior[3] = num_unknown_known.float() / num_unknown
    data_prior[4] = num_unknown_unlabel.float() / num_unknown
    data_prior[5] = num_unknown_class.float() / num_unknown

    data_class_prior = [0, 0, 0]
    data_class_prior[0] = num_known_class.float() / new_train_set.size()[0]
    data_class_prior[1] = num_unlabel_class.float() / new_train_set.size()[0]
    data_class_prior[2] = num_unknown_class.float() / new_train_set.size()[0]

    print('more data', num_unknown_class - num_unknown_unlabel)
    print(data_prior, data_class_prior,  num_known_class, num_unlabel_class, num_unknown_class)

    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P or test_label[i] == N or test_label[i] == A:
            index.append(i)

    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = new_test_label.clone()
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P:
            new_test_label[i] = 0
        if new_test_label_index[i] == N:
            new_test_label[i] = 1
        if new_test_label_index[i] == A:
            new_test_label[i] = 2

    print('data set  Parameter： unknown set 3 known set 2 unlabel set 6 number of unknown and unlabel  {} {}'.format(
        num_data / 2, num_data / 2))
    test_num_unknown = torch.eq(new_test_label, 2).sum()
    print('train_x shape:%s, train_y shape:%s' % (train_data.shape, train_label.shape))
    print('test_x shape:%s, test_y shape:%s' % (test_data.shape, test_label.shape))
    print('A class:{}, N class: {}, A class: {}'.format(P, N, A))
    print('PU number:{}, AC number: {}'.format(num_known + num_unlabel, num_unknown))
    print('Test number:{}, test AC number: {}'.format(new_test_label.size()[0], test_num_unknown))
    return new_train_set, new_train_label, new_test_set, new_test_label, data_prior, data_class_prior
