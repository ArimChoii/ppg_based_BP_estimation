from torch.utils.data import Dataset
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import numpy as np
import glob
import json

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.concat([pd.read_csv(data) for data in glob.glob(data_path + '/*.csv')], ignore_index=True)
        # min / max 찾고 list 선언 /
        # self.min = np.min([np.min(json.loads(x)) for x in self.data['label']])
        # self.max = np.max([np.max(json.loads(x)) for x in self.data['label']])

    def __len__(self):
        return len(self.data['ppg'])

    def __getitem__(self, idx):
        PPG = torch.FloatTensor(json.loads(self.data['ppg'][idx]))
        # plt.plot(PPG)
        # plt.xlabel('Time')
        # plt.ylabel('original')
        # plt.title('original ppg')
        # plt.legend()
        # plt.show()
        BP = torch.FloatTensor(json.loads(self.data['label'][idx]))
        SBP = BP[0].unsqueeze(0)
        DBP = BP[1].unsqueeze(0)
        # SBP = BP[0].item()
        # DBP = BP[1].item()
        # min_SBP = SBP
        # max_SBP = SBP
        # min_DBP = DBP
        # max_DBP = DBP
        return PPG, SBP, DBP  # , min_SBP, max_SBP, min_DBP, max_DBP

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
        # print('Filtered PPG: ', filtered_ppg)
        return scaled_ppg

    def validate_data(self, SBP, DBP):
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

