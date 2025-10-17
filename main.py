from torch.utils.data import TensorDataset, DataLoader
import torch
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

import train
import test
import minist_data
import SVHN_data
import Network
import cifar_10_data
import SVHN_data_balance
import time


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

def data_gen(new_train_set, new_train_label, new_test_set, new_test_label):
    train_dataset = TensorDataset(new_train_set, new_train_label)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = 256,
                              shuffle = True)

    test_dataset = TensorDataset(new_test_set, new_test_label)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size = 128,
                              shuffle=True)
    return train_loader, test_loader
def main():
    rato = 0.6
    parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default:64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default:1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default:1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--NC', type=int, default=2, metavar='N',
                        help='the number of negative classes')
    parser.add_argument('--data_set', type=int, default=1, metavar='N',
                        help='minist or fashion minist or Kuzushiji minist')
    parser.add_argument('--data_choice', type=int, default=3, metavar='N',
                        help='minist or SVHN or cifar')
    parser.add_argument('--model_choice', type=int, default=2, metavar='N',
                        help='MPU OR ESA or NMPU or NPU OR  PU OR CoMPU')
    parser.add_argument('--trials_choice', type=int, default=3, metavar='N',
                        help='number of iter')
    args = parser.parse_args()

    max_test_acc = torch.zeros(args.trials_choice)
    max_test_tpr = torch.zeros(args.trials_choice)
    for i_train_iter in range(args.trials_choice):
        torch.manual_seed(i_train_iter)
        np.random.seed(i_train_iter)
        if args.data_choice == 1:
            new_train_set, new_train_label, new_test_set, new_test_label, data_prior = minist_data.read_data_minist_MPU_rato(args.data_set, i_train_iter, rato)
            model = Network.Net()
            if i_train_iter == 0:
                parser.add_argument('--lr_known', type=float, default=8e-2, metavar='LR_m',
                                    help='learning rate for known(default:1.0)')
                parser.add_argument('--known_gamma', type=float, default=0.1, metavar='M',
                                    help='Learning rate step gamma (default:0.5)')
                parser.add_argument('--step_size', type=int, default=20, metavar='N',
                                    help='size')
                parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                    help='number of eopchs to train （default:14)')

        elif args.data_choice == 2:
            new_train_set, new_train_label, new_test_set, new_test_label, data_prior, data_class_prior = SVHN_data.read_data_SVHN(args.data_set)
            model = Network.CNN_Net_SVHN()
            if i_train_iter == 0:
                parser.add_argument('--lr_known', type=float, default=6e-1, metavar='LR_m',
                                    help='learning rate for known(default:1.0)')
                parser.add_argument('--known_gamma', type=float, default=0.1, metavar='M',
                                    help='Learning rate step gamma (default:0.5)')
                parser.add_argument('--step_size', type=int, default=8, metavar='N',
                                    help='size')

        elif args.data_choice == 3:
            new_train_set, new_train_label, new_test_set, new_test_label, data_prior = cifar_10_data.read_data_cifar_10_all()
            model = Network.CNN_Net_cifar().to(device)
            if i_train_iter == 0:
                parser.add_argument('--lr_known', type=float, default=8e-1, metavar='LR_m',
                                    help='learning rate for known(default:1.0)')
                parser.add_argument('--known_gamma', type=float, default=0.1, metavar='M',
                                    help='Learning rate step gamma (default:0.5)')
                parser.add_argument('--step_size', type=int, default=10, metavar='N',
                                    help='size')
                parser.add_argument('--epochs', type=int, default=50, metavar='N',
                                    help='number of eopchs to train （default:14)')
        elif args.data_choice == 4:
            new_train_set, new_train_label, new_test_set, new_test_label, data_prior, data_class_prior = SVHN_data_balance.read_data_MNIST(args.data_set, i_train_iter)
            model = Network.Net()
            if i_train_iter == 1:
                parser.add_argument('--lr_known', type=float, default=5e-4, metavar='LR_m',
                                    help='learning rate for known(default:1.0)')
                parser.add_argument('--known_gamma', type=float, default=0.1, metavar='M',
                                    help='Learning rate step gamma (default:0.5)')
                parser.add_argument('--step_size', type=int, default=10, metavar='N',
                                    help='size')

        args = parser.parse_args()
        optimizer_known = optim.Adadelta(model.parameters(), lr=args.lr_known)
        scheduler_known = StepLR(optimizer_known, step_size=args.step_size, gamma=args.known_gamma)

        print('data choice: {}\n 1:minist 2:SVHN 3:cifar \n mnist data set: {} \n 1: minist 2: fashion minsit 3: Kuzushiji_minist\n Train Parameter：step size {}, known gamma {}, known lr {}\n model chioce: {}'.format(
            args.data_choice, args.data_set, args.step_size, args.known_gamma,  args.lr_known, args.model_choice))

        train_loader, test_loader = data_gen(new_train_set, new_train_label, new_test_set, new_test_label)

        test_acc = torch.zeros(args.epochs)
        test_tpr = torch.zeros(args.epochs)
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train.train_multi_split(args, model, train_loader, optimizer_known, epoch, args.NC, data_prior, rato)
            test_acc[epoch-1], test_tpr[epoch-1] = test.test_multi(model,  test_loader, data_prior)
            scheduler_known.step()
            end = time.perf_counter()
            print('Running time: %s Seconds' % (end - start))
        max_test_acc[i_train_iter] = torch.max(test_acc, 0)[0]
        max_test_tpr[i_train_iter] = test_tpr[torch.max(test_acc, 0)[1]]
    acc_mean = torch.mean(max_test_acc)
    acc_std = torch.std(max_test_acc)
    tpr_mean = torch.mean(max_test_tpr)
    tpr_std = torch.std(max_test_tpr)
    print(max_test_acc, max_test_tpr)
    print(acc_mean, acc_std, tpr_mean, tpr_std)

if __name__ == '__main__':
    main()






