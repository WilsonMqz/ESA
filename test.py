import torch
import train
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

def test_multi( model,  test_loader, data_prior):
    model.eval()
    test_loss = 0
    correct = 0
    tp = 0
    new_class = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.tensor(data, dtype=torch.float32).to(device)
            target = target.to(device)
            output = model(data)
            test_loss += torch.mean(train.c_loss(output, target.long()))
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(target.view_as(pred).size()[0]):
                if target.view_as(pred)[i] == 9:
                    new_class = new_class + 1
                    if pred[i] == 9:
                        tp = tp + 1
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss = test_loss / len(test_loader)

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%), TPR: ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), 100.*tp / new_class))

    return 100. * correct / len(test_loader.dataset), 100.*tp / new_class

