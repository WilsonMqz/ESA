import os
import scipy.io as sio
import random
import numpy as np
import torchvision.datasets.mnist as mnist
import torch

def read_data_SVHN(data_set):
    num_data = 20000
    random.seed(1)

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
    num_known_known = 0
    num_unlabel_known = 0
    num_unknown_known = 0
    num_unknown_unknown = 0
    num_unknown_unlabel = 0
    num_unlabel_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == 5 or train_label[i] == 4 or train_label[i] == 3:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = new_train_label.clone()
    rand_index = random.sample(range(new_train_set.size()[0]), new_train_set.size()[0])
    for i in range(new_train_set.size()[0]):
        if i <= new_train_set.size()[0]/2:
            new_train_label[rand_index[i]] = 2
            if new_train_label_index[rand_index[i]] == 5:
                num_unknown_known = num_unknown_known + 1
            if new_train_label_index[rand_index[i]] == 4:
                num_unknown_unlabel = num_unknown_unlabel + 1
            if new_train_label_index[rand_index[i]] == 3:
                num_unknown_unknown = num_unknown_unknown + 1
        elif i <= 3*new_train_set.size()[0]/4:
            if new_train_label_index[rand_index[i]] == 5:
                new_train_label[rand_index[i]] = 0
                num_known_known = num_known_known + 1
            elif new_train_label_index[rand_index[i]] == 4:
                new_train_label[rand_index[i]] = 1
                num_unlabel_unlabel = num_unlabel_unlabel + 1
        else:
            if new_train_label_index[rand_index[i]] == 5:
                num_unlabel_known = num_unlabel_known + 1
                new_train_label[rand_index[i]] = 1
            if new_train_label_index[rand_index[i]] == 4:
                num_unlabel_unlabel = num_unlabel_unlabel + 1
                new_train_label[rand_index[i]] = 1

    num_known = torch.eq(new_train_label, 0).sum()
    num_unlabel = torch.eq(new_train_label, 1).sum()
    num_unknown = torch.eq(new_train_label, 2).sum()
    num_known_class = torch.eq(train_label, 5).sum()
    num_unlabel_class = torch.eq(train_label, 4).sum()
    num_unknown_class = torch.eq(train_label, 3).sum()

    num_unlabel_known = torch.tensor(num_unlabel_known)
    num_unlabel_unlabel = torch.tensor(num_unlabel_unlabel)
    num_unknown_known = torch.tensor(num_unknown_known)
    num_unknown_unlabel = torch.tensor(num_unknown_unlabel)
    num_unknown_unknown = torch.tensor(num_unknown_unknown)


    data_prior[0] = torch.tensor(1)
    data_prior[1] = num_unlabel_known.float() / num_unlabel
    data_prior[2] = num_unlabel_unlabel.float() / num_unlabel
    data_prior[3] = num_unknown_known.float() / num_unknown
    data_prior[4] = num_unknown_unlabel.float() / num_unknown
    data_prior[5] = num_unknown_unknown.float() / num_unknown

    data_class_prior = [0, 0, 0]
    data_class_prior[0] = num_known_class.float() / new_train_set.size()[0]
    data_class_prior[1] = num_unlabel_class.float() / new_train_set.size()[0]
    data_class_prior[2] = num_unknown_class.float() / new_train_set.size()[0]

    index = []
    for i in range(new_train_label.size()[0]):
        if new_train_label[i] == 0 or new_train_label[i] == 1 or new_train_label[i] == 2:
            index.append(i)
    index = torch.tensor(index)
    new_train_set_t = torch.index_select(new_train_set, dim=0, index=index)
    new_train_label_t = torch.index_select(new_train_label, dim=0, index=index)

    print('more data', num_unknown_class - num_unknown_unlabel)

    print(data_prior, data_class_prior,  num_known_class, num_unlabel_class, num_unknown_class)

    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == 5 or test_label[i] == 4 or test_label[i] == 3:
            index.append(i)

    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = new_test_label.clone()
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == 5:
            new_test_label[i] = 0
        if new_test_label_index[i] == 4:
            new_test_label[i] = 1
        if new_test_label_index[i] == 3:
            new_test_label[i] = 2

    print('data set  Parameter： unknown set 3 known set 2 unlabel set 6 number of unknown and unlabel  {} {}'.format(
        num_data / 2, num_data / 2))

    print('train_x shape:%s, train_y shape:%s' % (train_data.shape, train_label.shape))
    print('test_x shape:%s, test_y shape:%s' % (test_data.shape, test_label.shape))
    return new_train_set_t, new_train_label_t, new_test_set, new_test_label, data_prior, data_class_prior

def read_data_MNIST(data_set,i_train_iter):
    num_data = 25000
    random.seed(i_train_iter)
    root = "D:/code/data/minist/minist/"
    train_set = (
        mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
    )
    test_set = (
        mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
    )

    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])

    data_prior = [0, 0, 0, 0, 0, 0]
    num_known_known = 0
    num_unlabel_known = 0
    num_unknown_known = 0
    num_unknown_unknown = 0
    num_unknown_unlabel = 0
    num_unlabel_unlabel = 0
    index = []
    for i in range(train_label.size()[0]):
        if train_label[i] == 5 or train_label[i] == 4 or train_label[i] == 3:
            index.append(i)
    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = new_train_label.clone()
    rand_index = random.sample(range(new_train_set.size()[0]), new_train_set.size()[0])
    for i in range(new_train_set.size()[0]):
        if i <= new_train_set.size()[0] / 2:
            new_train_label[rand_index[i]] = 2
            if new_train_label_index[rand_index[i]] == 5:
                num_unknown_known = num_unknown_known + 1
            if new_train_label_index[rand_index[i]] == 4:
                num_unknown_unlabel = num_unknown_unlabel + 1
            if new_train_label_index[rand_index[i]] == 3:
                num_unknown_unknown = num_unknown_unknown + 1
        elif i <= 3 * new_train_set.size()[0] / 4:
            if new_train_label_index[rand_index[i]] == 5:
                new_train_label[rand_index[i]] = 0
                num_known_known = num_known_known + 1
            elif new_train_label_index[rand_index[i]] == 4:
                new_train_label[rand_index[i]] = 1
                num_unlabel_unlabel = num_unlabel_unlabel + 1
        else:
            if new_train_label_index[rand_index[i]] == 5:
                num_unlabel_known = num_unlabel_known + 1
                new_train_label[rand_index[i]] = 1
            if new_train_label_index[rand_index[i]] == 4:
                num_unlabel_unlabel = num_unlabel_unlabel + 1
                new_train_label[rand_index[i]] = 1
    num_known = torch.eq(new_train_label, 0).sum()
    num_unlabel = torch.eq(new_train_label, 1).sum()
    num_unknown = torch.eq(new_train_label, 2).sum()
    num_known_class = torch.eq(train_label, 5).sum()
    num_unlabel_class = torch.eq(train_label, 4).sum()
    num_unknown_class = torch.eq(train_label, 3).sum()

    num_unlabel_known = torch.tensor(num_unlabel_known)
    num_unlabel_unlabel = torch.tensor(num_unlabel_unlabel)
    num_unknown_known = torch.tensor(num_unknown_known)
    num_unknown_unlabel = torch.tensor(num_unknown_unlabel)
    num_unknown_unknown = torch.tensor(num_unknown_unknown)

    num_data = num_known + num_unlabel + num_unknown
    data_prior[0] = torch.tensor(1).float()
    data_prior[1] = num_unlabel_known.float() / num_unlabel
    data_prior[2] = num_unlabel_unlabel.float() / num_unlabel
    data_prior[3] = num_unknown_known.float() / num_unknown
    data_prior[4] = num_unknown_unlabel.float() / num_unknown
    data_prior[5] = num_unknown_unknown.float() / num_unknown

    data_class_prior = [0, 0, 0]
    data_class_prior[0] = num_known_class.float() / new_train_set.size()[0]
    data_class_prior[1] = num_unlabel_class.float() / new_train_set.size()[0]
    data_class_prior[2] = num_unknown_class.float() / new_train_set.size()[0]

    index = []
    for i in range(new_train_label.size()[0]):
        if new_train_label[i] == 0 or new_train_label[i] == 1 or new_train_label[i] == 2:
            index.append(i)
    index = torch.tensor(index)
    new_train_set_t = torch.index_select(new_train_set, dim=0, index=index)
    new_train_label_t = torch.index_select(new_train_label, dim=0, index=index)
    new_train_label_index_t = torch.index_select(new_train_label_index, dim=0, index=index)

    num_known_class = torch.eq(new_train_label_index_t, 5).sum()
    num_unlabel_class = torch.eq(new_train_label_index_t, 4).sum()
    num_unknown_class = torch.eq(new_train_label_index_t, 3).sum()


    data_class_prior = [0, 0, 0]
    data_class_prior[0] = num_known_class.float() / new_train_set_t.size()[0]
    data_class_prior[1] = num_unlabel_class.float() / new_train_set_t.size()[0]
    data_class_prior[2] = num_unknown_class.float() / new_train_set_t.size()[0]

    print('more data', num_unknown_class - num_unknown_unlabel)

    print(data_prior, data_class_prior, num_known_class, num_unlabel_class, num_unknown_class)

    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == 5 or test_label[i] == 4 or test_label[i] == 3:
            index.append(i)

    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = new_test_label.clone()
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == 5:
            new_test_label[i] = 0
        if new_test_label_index[i] == 4:
            new_test_label[i] = 1
        if new_test_label_index[i] == 3:
            new_test_label[i] = 2

        test_ratio_known = 0.4

        test_ratio_unlabel = 0.01
        test_ratio_unknown = 0.005
        test_num_known = test_ratio_known * new_test_label.size()[0]
        test_num_unlabel = test_ratio_unlabel * new_test_label.size()[0]
        test_num_unknown = test_ratio_unknown * new_test_label.size()[0]
        index = []
        ite_known = 0
        ite_unlabel = 0
        ite_unknown = 0
        for i in range(new_test_label_index.size()[0]):
            if new_test_label[i] == 0 and ite_known <= test_num_known:
                index.append(i)
                ite_known = ite_known + 1
            if new_test_label[i] == 1 and ite_unlabel <= test_num_unlabel:
                index.append(i)
                ite_unlabel = ite_unlabel + 1
            if new_test_label[i] == 2 and ite_unknown <= test_num_unknown:
                index.append(i)
                ite_unknown = ite_unknown + 1
        index = torch.tensor(index)
        new_test_set_t = torch.index_select(new_test_set, dim=0, index=index)
        new_test_label_t = torch.index_select(new_test_label, dim=0, index=index)

    print('data set  Parameter： unknown set 3 known set 2 unlabel set 6 number of unknown and unlabel  {} {}'.format(
        num_data / 2, num_data / 2))

    print('train_x shape:%s, train_y shape:%s' % (train_data.shape, train_label.shape))
    print('test_x shape:%s, test_y shape:%s' % (test_data.shape, test_label.shape))
    return new_train_set_t.unsqueeze(1), new_train_label_t, new_test_set_t.unsqueeze(1), new_test_label_t, data_prior, data_class_prior