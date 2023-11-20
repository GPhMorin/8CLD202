""" Test the model on the test set. """
# Source: https://github.com/pytorch/examples/blob/main/mnist/main.py

import torch
import torch.nn.functional as F

def test(model, test_set):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_set:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_set.dataset)

    print('\nTest: Average loss {:.4f} Accuracy {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_set.dataset),
        100. * correct / len(test_set.dataset)))