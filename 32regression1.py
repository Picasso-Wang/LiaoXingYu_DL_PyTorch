import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[0.3], [4.4], [5.5], [6.71], [6.9], [4.1], [9.7], [6.2],
                    [7.6], [2.2], [7], [10.9], [5], [8], [3]], dtype=np.float32)
y_train = np.array([[0.7], [3.7], [5], [7], [7.7], [4.57], [9.4], [6.5],
                    [7.2], [2.8], [7.5], [10.6], [5.9], [8.3], [3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)        # input and output is one dimension

    def forward(self, x):
        out = self.linear(x)
        return out


if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 100
if torch.cuda.is_available():
    inputs = Variable(x_train).cuda()
    target = Variable(y_train).cuda()
else:
    inputs = Variable(x_train)
    target = Variable(y_train)
for epoch in range(epochs):
    # if torch.cuda.is_available():
    #     inputs = Variable(x_train).cuda()
    #     target = Variable(y_train).cuda()
    # else:
    #     inputs = Variable(x_train)
    #     target = Variable(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        plt.cla()  # Clear the current axes.
        plt.scatter(x_train.data.numpy(), y_train.data.numpy())
        plt.plot(x_train.data.numpy(), out.data.numpy())
        plt.text(0.5, 0.5, 'Loss={:.4f}'.format(loss.data), fontdict={'size': 10, 'color': 'red'})
        # print('Epoch [{} / {}], loss: {:.6f}'.format(epoch, epochs, loss.data))
        plt.pause(0.1)




