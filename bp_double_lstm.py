import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
from sklearn.metrics import mean_absolute_error
import os
from math import floor

class LSTMDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        try:
            self.data = pd.read_csv(csv_file)
            if 'ppg' not in self.data.columns or 'label' not in self.data.columns:
                raise ValueError("CSV file must contain 'ppg' and 'label' columns.")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            self.data = pd.DataFrame(columns=['ppg', 'label'])

        self.seq_length = seq_length
        self.scaler = StandardScaler()
        self.filtered_ppgs = self.filtering()
        self.preprocess = self.total_preprocessing()
        print(f"Number of valid samples in {os.path.basename(csv_file)}: {len(self.preprocess)}")

    def filtering(self):
        fs = 125
        lowcut = 0.5
        highcut = 8
        order = 4
        b, a = butter(order, [lowcut, highcut], 'bandpass', fs=fs)
        filtered_ppgs = []

        for idx, ppg in enumerate(self.data['ppg']):
            try:
                ppg = np.array(json.loads(ppg))
                if len(ppg) != 875:
                    raise ValueError(f"PPG data is not 875 samples: {len(ppg)} samples")
                filtered_ppg = filtfilt(b, a, ppg)
                scaled_ppg = self.scaler.fit_transform(filtered_ppg.reshape(-1, 1)).flatten()
                filtered_ppgs.append(scaled_ppg)
            except Exception as e:
                print(f"Error processing PPG data at index {idx}: {e}")
                filtered_ppgs.append(np.zeros(self.seq_length))  # 빈 데이터를 기본값으로 추가

        return filtered_ppgs

    def validate_and_replace(self, SBP, DBP):
        SBP_min = 40
        SBP_max = 200
        DBP_min = 40
        DBP_max = 120

        # 유효 범위로 SBP 및 DBP 대체
        valid_sbp = np.clip(SBP, SBP_min, SBP_max)
        valid_dbp = np.clip(DBP, DBP_min, DBP_max)

        return valid_sbp, valid_dbp

    def total_preprocessing(self):
        preprocess = []
        for idx in range(len(self.data)):
            try:
                BP = torch.FloatTensor(json.loads(self.data['label'][idx]))
                SBP = BP[0]
                DBP = BP[1]

                valid_sbp, valid_dbp = self.validate_and_replace(SBP, DBP)

                if valid_sbp != SBP or valid_dbp != DBP:
                    print(f"Replaced invalid SBP/DBP values at index {idx}: SBP={SBP}, DBP={DBP} -> SBP={valid_sbp}, DBP={valid_dbp}")

                preprocess.append((idx, valid_sbp, valid_dbp))
            except Exception as e:
                print(f"Error processing label data at index {idx}: {e}")
        return preprocess

    def __len__(self):
        return len(self.preprocess)

    def __getitem__(self, idx):
        valid_idx, valid_sbp, valid_dbp = self.preprocess[idx]
        ppg_seq = self.filtered_ppgs[valid_idx]

        # 시퀀스 길이 확인
        if len(ppg_seq) != self.seq_length:
            raise ValueError(f"PPG sequence length is incorrect at index {valid_idx}. Expected {self.seq_length}, but got {len(ppg_seq)}")

        ppg_seq = torch.FloatTensor(ppg_seq)
        return ppg_seq.view(self.seq_length, 1), valid_sbp, valid_dbp  # [seq_length, 1]

class DoubleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length):
        super(DoubleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        # nn.LSTM, pytorch의 내장 클래스
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * seq_length, output_size)

    def forward(self, x):
        h0_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out1, _ = self.lstm1(x, (h0_1, c0_1))

        h0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out2, _ = self.lstm2(out1, (h0_2, c0_2))
        out = out2.contiguous().view(x.size(0), -1)  # Flatten LSTM output
        out = self.fc(out)
        return out

def loss_function(x, y):
    loss = nn.L1Loss()  # MAE 사용
    return loss(x, y)

def train(epoch, model, optimizer, loss_function, dataloader):
    model.train()
    train_loss = 0
    for batch_idx, (PPG, SBP, DBP) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(PPG)
        loss = loss_function(pred, torch.stack([SBP, DBP], dim=1))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 875 == 0:
            print('Train Epoch: {}/{} \tLoss: {:.6f}'.format(epoch, epochs, loss.item()))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / (batch_idx + 1)))
    return train_loss / (batch_idx + 1)

def test(model, dataloader, loss_function):
    model.eval()
    pred_list = []
    SBP_list = []
    DBP_list = []
    test_loss_sum = 0
    with torch.no_grad():
        for batch_idx, (PPG, SBP, DBP) in enumerate(dataloader):
            pred = model(PPG)
            test_loss = loss_function(pred, torch.stack([SBP, DBP], dim=1))
            test_loss_sum += test_loss.item()

            pred_list.append(pred.numpy())
            SBP_list.append(SBP.numpy())
            DBP_list.append(DBP.numpy())

    test_loss_sum /= (batch_idx + 1)
    print('====> Test set loss: {:.4f}'.format(test_loss_sum))

    # Flatten lists
    pred_list = np.vstack(pred_list)
    SBP_list = np.hstack(SBP_list)
    DBP_list = np.hstack(DBP_list)

    # MAE Score
    mae_sbp = mean_absolute_error(SBP_list, pred_list[:, 0])
    mae_dbp = mean_absolute_error(DBP_list, pred_list[:, 1])

    print(f'SBP - MAE: {mae_sbp:.4f}')
    print(f'DBP - MAE: {mae_dbp:.4f}')

    return test_loss_sum

# Tkinter를 사용하여 데이터 경로를 선택
root = Tk()
root.title('Data Path')
root.data_path = filedialog.askdirectory()
root.withdraw()  # Tkinter 창 숨김

# 모든 CSV 파일을 하나씩 데이터셋으로 만듦
seq_length = 875  # 7초 전체 사용
csv_files = glob.glob(root.data_path + '/*.csv')

train_file_count = floor(len(csv_files) * 0.8)
test_file_count = len(csv_files) - train_file_count

train_files = csv_files[:train_file_count]
test_files = csv_files[train_file_count:]

# 각 파일을 데이터셋으로 만들기
train_datasets = [LSTMDataset(csv_file, seq_length=seq_length) for csv_file in train_files]
test_datasets = [LSTMDataset(csv_file, seq_length=seq_length) for csv_file in test_files]

# 데이터셋을 결합
train_dataset = ConcatDataset(train_datasets)
test_dataset = ConcatDataset(test_datasets)

# DataLoader 생성
batch_size = 125
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 설정
model = DoubleLSTM(input_size=1, hidden_size=64, num_layers=3, output_size=2, seq_length=seq_length)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 30

# 학습 및 손실 함수 초기화
loss_function = nn.L1Loss()
loss_values = []

for epoch in range(1, epochs + 1):
    Loss = train(epoch, model, optimizer, loss_function, train_dataloader)
    loss_values.append(Loss)

plt.plot(loss_values, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

test_loss = test(model, test_dataloader, loss_function)