import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import matplotlib.pyplot as plt


def make_features(x):
    ''''Builds features, i.e.  matrix with colunms [x, x^2, x^3]'''
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1, 4)], 1)


W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def f(x):
    return x.mm(W_target) + b_target[0]


def getBatch(batch_size=32):
    '''Builds a batch i.e. (x ,f(x)) pair.'''
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y), random


# Define model
class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
x_test = torch.linspace(-2, 2, 101)
y_test = f(make_features(x_test))
while True:
    # Get data
    batch_x, batch_y, x_random = getBatch()
    # Forward pass
    output = model(batch_x)
    loss = criterion(output, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch: {:>3}, loss: {:.3f}'.format(epoch, loss.data))
    print('weight: ', list(model.parameters())[0])
    epoch += 1
    if loss.data <= 1e-3 or epoch > 200:
        break

    if epoch % 10 == 0:
        # predict
        model.eval()
        predict = model(Variable(make_features(x_test)))
        plt.cla()
        plt.plot(x_test.numpy(), y_test.numpy(), c='red', label='target')
        plt.plot(x_test.numpy(), predict.data.numpy(), c='green', label='predict')
        plt.legend(loc='best')
        plt.text(0.1, 0.1, 'Loss={:.4f}'.format(loss.data), fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.3)

        # plt.cla()
        # plt.scatter(x_random.data.numpy(), batch_y.data.numpy(), c='red')
        # plt.scatter(x_random.data.numpy(), output.data.numpy(), c='green')
        # plt.text(0.5, 0.5, 'Loss={:.4f}'.format(loss.data), fontdict={'size': 10, 'color': 'red'})
        # plt.pause(0.5)





