import torch
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

df = pd.read_csv("winequality.csv")
df = df.dropna()
df["type"] = df['type'].replace(['white', 'red'], [0, 1])

X = df.iloc[:, 0:-1]
y = df["quality"]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=42)

ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)

Xtrain = torch.from_numpy(Xtrain.astype(np.float32))
Ytrain = torch.from_numpy(Ytrain.to_numpy().astype(np.float32)).view(-1, 1)

Xtest = torch.from_numpy(Xtest.astype(np.float32))
Ytest = torch.from_numpy(Ytest.to_numpy().astype(np.float32)).view(-1, 1)


class wineDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


dataset = wineDataset(Xtrain, Ytrain)

dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)


class model(nn.Module):
    def __init__(self, size):
        super(model, self).__init__()
        self.linear1 = nn.Linear(size, 64)
        # self.relu1=nn.ReLU()
        self.relu1 = nn.PReLU()
        self.linear2 = nn.Linear(64, 32)
        # self.relu2 = nn.ReLU()
        self.relu2 = nn.PReLU()
        self.linear3 = nn.Linear(32, 1)

    def forward(self, x):
        A = self.linear1(x)
        A = self.relu1(A)
        A = self.linear2(A)
        A = self.relu2(A)
        A = self.linear3(A)

        return A


size = Xtrain.shape[1]
m = model(size)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(m.parameters(), lr=0.001)

losses = []

for epoch in range(100):

    for X, Y in dataLoader:
        ypred = m(X)
        l = loss(ypred, Y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"epoch= {epoch + 1}, loss= {l.item()}")
    losses.append(l)

with torch.no_grad():
    pred = m(Xtest)
    Tloss = nn.MSELoss()
    tloss = Tloss(pred, Ytest)
    print(f"Test loss: {tloss.item()}")
    plt.plot(range(1, 101), losses)
    plt.show()
