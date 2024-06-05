import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

class LSTMDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        try:
            self.data = pd.read_csv(csv_file)
            if 'Pulse Wave' not in self.data.columns:
                raise ValueError("CSV file must contain 'Pulse Wave' column.")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            self.data = pd.DataFrame(columns=['Pulse Wave'])

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

        ppg = self.data['Pulse Wave'].values.astype(float)
        for start in range(0, len(ppg) - self.seq_length + 1):
            segment = ppg[start:start + self.seq_length]
            filtered_ppg = filtfilt(b, a, segment)
            scaled_ppg = self.scaler.fit_transform(filtered_ppg.reshape(-1, 1)).flatten()
            filtered_ppgs.append(scaled_ppg)

        return filtered_ppgs

    def total_preprocessing(self):
        # Dummy labels as placeholders, replace with actual labels if available
        preprocess = [(idx, 0, 0) for idx in range(len(self.filtered_ppgs))]
        return preprocess

    def __len__(self):
        return len(self.preprocess)

    def __getitem__(self, idx):
        valid_idx, valid_sbp, valid_dbp = self.preprocess[idx]
        ppg_seq = self.filtered_ppgs[valid_idx]

        if len(ppg_seq) != self.seq_length:
            raise ValueError(f"PPG sequence length is incorrect at index {valid_idx}. Expected {self.seq_length}, but got {len(ppg_seq)}")

        ppg_seq = torch.FloatTensor(ppg_seq)
        return ppg_seq.view(self.seq_length, 1), valid_sbp, valid_dbp  # [seq_length, 1]

class DoubleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, output_size, seq_length):
        super(DoubleLSTM, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size2 * seq_length, output_size)

    def forward(self, x):
        h0_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(device)
        c0_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(device)
        out1, _ = self.lstm1(x, (h0_1, c0_1))

        h0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(device)
        c0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(device)
        out2, _ = self.lstm2(out1, (h0_2, c0_2))
        out = out2.contiguous().view(x.size(0), -1)  # Flatten LSTM output
        out = self.fc(out)
        return out

# 장치 설정 (GPU 사용 가능 시)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의 및 로드
model = DoubleLSTM(input_size=1, hidden_size1=32, hidden_size2=32, num_layers=3, output_size=2, seq_length=875).to(device)
model.load_state_dict(torch.load('final_model.pth'))
model.eval()

# 데이터셋 및 데이터로더 준비
csv_file = '_Ubpulse-Pulse-Wave.csv'  # 테스트할 CSV 파일 경로
seq_length = 875
dataset = LSTMDataset(csv_file, seq_length)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 모델 예측
predictions = []
for i, (ppg_seq, sbp, dbp) in enumerate(dataloader):
    ppg_seq = ppg_seq.to(device)
    output = model(ppg_seq)
    predictions.append(output.cpu().detach().numpy())

predictions = np.array(predictions)
print("Predictions:")
print(predictions)
print("Mean of Predictions:")
print(np.mean(predictions, axis=0))