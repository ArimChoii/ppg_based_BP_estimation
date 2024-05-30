import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tkinter import filedialog
from tkinter import *
import json
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
#import alexnet
#from alexnet import Alexnet

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.concat([pd.read_csv(data) for data in glob.glob(data_path + '/*.csv')], ignore_index=True)

    def __len__(self):
        return len(self.data['ppg'])

    def __getitem__(self, idx):
        PPG = torch.FloatTensor(json.loads(self.data['ppg'][idx]))
        BP = torch.FloatTensor(json.loads(self.data['label'][idx]))
        SBP = BP[0]
        DBP = BP[1]
        return PPG, SBP, DBP

    def filtering(self, PPG):
        fs = 125
        lowcut = 0.5
        highcut = 8
        order = 4
        # 4th order butterworth filter for PPG preprcessing
        b, a = butter(order, [lowcut, highcut], 'bandpass', fs=fs)
        filtered_ppg = filtfilt(b, a, PPG)
        scaler = StandardScaler()
        scaled_ppg = scaler.fit_transform(filtered_ppg.reshape(-1, 1)).flatten()
        return scaled_ppg

    def validate_date(self, SBP, DBP):
        NN = [1 / (x / 125) for x in np.diff(np.where(SBP > 0)[0])]
        HR = 60 / np.mean(NN)
        RR_min = 0.3
        RR_max = 1.4
        SBP_min = 40
        SBP_max = 200
        DBP_min = 40
        DBP_max = 120

        if HR < 50 or HR > 140:
            return False

        # Check if RR interval is in the range of 0.3 to 1.4 seconds
        if any(np.diff(np.where(SBP > 0)[0]) / 125 < RR_min) or any(np.diff(np.where(SBP > 0)[0]) / 125 > RR_max):
            return False

        # Check if SBP and DBP values are within reasonable ranges
        if any(SBP < SBP_min) or any(SBP > SBP_max) or any(DBP < DBP_min) or any(DBP > DBP_max):
            return False

        return True

    def process_data(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        PPG = df['PPG'].values
        SBP = df['SBP'].values
        DBP = df['DBP'].values

        # Perform data validation
        if not self.validate_data(PPG, SBP, DBP):
            return None

        # Call the filtering function
        scaled_ppg = self.filtering(PPG)
        print('scaled_ppg :' , scaled_ppg)
        # Return the preprocessed data
        return scaled_ppg


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.FC1 = nn.Sequential(nn.Linear(875, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU())
        self.FC2 = nn.Sequential(nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU())
        self.FC3 = nn.Sequential(nn.Linear(256, 128),
                                 nn.BatchNorm1d(128),
                                 nn.Sigmoid())
        self.FC4 = nn.Sequential(nn.Linear(128, 2))


    def forward(self, x):
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x = self.FC4(x)

        return x

root = Tk()
root.title('Data_path')
root.data_path = filedialog.askdirectory()
model = MLP()
optimizer = torch.optim.Adam(model.parameters())

dataset = CustomDataset(root.data_path)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=False)
def loss_function(x, y):
    loss = nn.L1Loss() # abs(original-prediction)
    Loss = loss(x, y)
    return Loss

def train(epoch):
    epochs = 50
    model.train()
    train_loss = 0
    for batch_idx, (PPG, SBP, DBP) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(PPG)
        loss = loss_function(pred, torch.stack([SBP, DBP], dim=1))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {}/{} \tLoss: {:.6f}'.format(epoch, epochs, loss.item()))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / (batch_idx + 1)))
    return train_loss / (batch_idx + 1)

def test():
    epochs = 10
    model.eval()
    test_loss_sum = 0
    with torch.no_grad():
        for batch_idx, (PPG, SBP, DBP) in enumerate(dataloader):
            pred = model(PPG)
            test_loss = loss_function(pred, torch.stack([SBP, DBP], dim=1))
            test_loss_sum += test_loss.item()

    test_loss_sum /= (batch_idx + 1)
    print('====> Test set loss: {:.4f}'.format(test_loss_sum))
    return test_loss_sum

epochs = 50
loss_values=[]

for epoch in range(1, epochs+1):
    Loss = train(epoch)
    loss_values.append(Loss)

plt.plot(loss_values, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

test_loss = test()