import torch
import torch.nn.functional as F

torch.manual_seed(1)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

def c_loss(output, label):
    class_mun = 10
    one_hot = F.one_hot(label.to(torch.int64), class_mun).to(device) * 2 - 1

    sig_out = output * one_hot
    y_label = torch.ones(sig_out.size()).to(device)
    out = torch.pow(sig_out-y_label, 2).to(device)

    return out

def MPU_loss_sevi_all(output, target, data_prior, rato):

    index_KN = []
    index_UN = []
    for i in range(target.size()[0]):
        if target[i] == 9:
            index_UN.append(i)
        else:
            index_KN.append(i)

    index_KN = torch.tensor(index_KN).long().to(device)
    index_UN = torch.tensor(index_UN).long().to(device)

    target_KN = torch.index_select(target, dim=0, index=index_KN).to(device)
    target_UN = torch.index_select(target, dim=0, index=index_UN).to(device)
    target_KN_AC = torch.ones(target_KN.size()).to(device)* 9
    output_KN = torch.index_select(output, dim=0, index=index_KN).to(device)
    output_UN = torch.index_select(output, dim=0, index=index_UN).to(device)

    loss_KN = c_loss(output_KN, target_KN)
    loss_KN_UN = c_loss(output_KN, target_KN_AC)
    loss_UN = c_loss(output_UN, target_UN)

    data_sum_prior = 1-data_prior[9]
    loss = data_sum_prior * (loss_KN - loss_KN_UN)
    loss_index = loss[:, 0]
    for i in range(1, 10):
        loss_index = loss_index + loss[:, i]

    index_loss_kn_se = torch.nonzero(loss_index > -3.5, as_tuple=False)

    loss_KN_SE = torch.index_select(loss, dim=0, index=torch.unique(index_loss_kn_se, sorted=False))

    loss_UN_index = loss_UN[:, 0]
    for i in range(1, 10):
        loss_UN_index = loss_UN_index + loss_UN[:, i]
    index_loss_UN = torch.nonzero(loss_UN_index > 0.02, as_tuple=False)#kmnist 0.08 # cifar 0.1

    loss_UN_SE = torch.index_select(loss_UN, dim=0, index=torch.unique(index_loss_UN, sorted=False))

    loss = torch.mean(loss_KN_SE) + torch.mean(loss_UN_SE)

    return loss
def train_multi_split(args, model, train_loader, optimizer_kn, epoch, NC, data_prior, rato):
    model.train().to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.tensor(data, dtype=torch.float32).to(device)

        optimizer_kn.zero_grad()
        output = model(data).to(device)
        loss_unsevi = MPU_loss_sevi_all(output, target.to(device), data_prior, rato)
        loss_unsevi.backward()
        optimizer_kn.step()
    print('Train sieve Epoch: {} '.format(epoch))
