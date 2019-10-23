import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

f = open('classificationDataSet.txt')
data = []
for line in f.readlines():
    tmp = line[:-1].split('\t')
    for i in range(3):
        tmp[i] = float(tmp[i])
    data.append(tmp)

data_x = [i[:2] for i in data]
data_x = torch.tensor(data_x)
data_y = [i[-1] for i in data]
data_y = torch.tensor(data_y)

draw_data = 1
if draw_data:
    x0 = list(filter(lambda x: x[-1] == 0.0, data))
    x1 = list(filter(lambda x: x[-1] == 1.0, data))

    x0_0 = [i[0] for i in x0]
    x0_1 = [i[1] for i in x0]
    x1_0 = [i[0] for i in x1]
    x1_1 = [i[1] for i in x1]

    plt.scatter(x0_0, x0_1, c='red', label='x_0')
    plt.scatter(x1_0, x1_1, c='green', label='x_1')
    plt.legend(loc='best')


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.ln = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.ln(x)
        x = self.sm(x)
        return x


model = LogisticRegression()
if torch.cuda.is_available():
    model.cuda()

criterion = nn.BCELoss()   # Binary Cross Entropy
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

if torch.cuda.is_available():
    x = Variable(data_x).cuda()
    y = Variable(data_y).cuda()
else:
    x = Variable(data_x)
    y = Variable(data_y)

for epoch in range(5000):
    # -------------forward--------------------------
    out = model(x)
    loss = criterion(out, y)
    mask = out.ge(0.5).float()
    mask = mask.squeeze()
    correct = (mask == y).sum()
    acc = correct.data.numpy() / x.size(0)

    # ------------backward--------------------------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print('epoch: {:>4}   acc: {:.2f}   loss: {:.3f}'.format(epoch, acc, loss.data))
        w0, w1 = model.ln.weight[0].data.numpy()
        b = float(model.ln.bias[0].data.numpy())
        plot_x = np.arange(-3, 3, 0.01)
        plot_y = (-w0*plot_x - b)/w1
        plt.plot(plot_x, plot_y)
        plt.pause(0.1)
        plt.show()



















