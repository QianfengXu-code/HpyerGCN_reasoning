import torch

threshold_km = 10
threshold_pearson = 0.9
SEQ_LEN = 12  # 输入序列长度（12小时）
PRED_LEN = 1  # 预测长度（1小时）
BATCH_SIZE = 32
EPOCHS = 50  # 增加最大epoch数，让早停机制发挥作用
HIDDEN_DIM = 64     # 64
GCN_OUT = 32    # 32
staion_dim = 32
time_dim = 6
hyper_dim = 6
LR = 0.001
PATIENCE = 10  # 早停耐心值（连续多少个epoch验证损失没有改善）
curvature = 0.2
tla_loss_weight = -1
lstm_loss_weight = 1

lstm_hidden = 4
Transformer_hidden = 5

model_name = 'Hyper_reason'   # ['Hyper_reason', 'GC_LSTM', 'LSTM', 'Transformer']

need_preprocess = False
need_reasoning = True
need_hyper = True

need_save = True

model_save_dir = './checkpoints/'
dir_root = './data2'
data_name = '/data_tensor.npy'
station_name = '/station_to_idx.joblib'
scaler_name = '/scalers.joblib'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

