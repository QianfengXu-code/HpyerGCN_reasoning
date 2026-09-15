from torch.utils.data import Dataset
from utils import *

class AirQualityDataset(Dataset):
    def __init__(self, data, seq_len=12, pred_len=1):
        self.data = data  # [总时间步, 节点数, 特征数]
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # 输入序列 [seq_len, nodes, features]
        # y_so2 = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, :, 1]
        # y_pm_10 = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, :, 5]
        # y_pm_25 = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, :, 6]  # PM2.5浓度 [pred_len, nodes]

        y_so2 = self.data[idx + self.seq_len + self.pred_len, :, 1].reshape(1, -1)
        y_pm_10 = self.data[idx + self.seq_len + self.pred_len, :, 5].reshape(1, -1)
        y_pm_25 = self.data[idx + self.seq_len + self.pred_len, :, 6].reshape(1, -1)  # PM2.5浓度 [pred_len, nodes]

        y = np.concatenate([y_so2, y_pm_10, y_pm_25], axis=0)
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
