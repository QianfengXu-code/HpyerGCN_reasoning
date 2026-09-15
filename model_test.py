import torch
import os
import config as cfg
import torch.nn as nn
from joblib import load
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models import *
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import logging
import time


def model_save(my_model, pth):

    torch.save(my_model.state_dict(), pth)


def metrics_gen():

    train_loader, val_loader, test_loader = dataloader_build(data_tensor, cfg)

    criterion = nn.MSELoss()

    model.eval()

    idx_0_preds = []
    idx_0_targets = []

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)

            if cfg.model_name == 'LSTM' or cfg.model_name == 'Transformer':
                pm25 = model(x)
                loss = criterion(pm25, y[:, 2].squeeze(1))

            else:

                so2, pm10, pm25, loss_tla = model(x, edge_index_distance, edge_index_pearson, graph_index_distance,
                                                  graph_index_pearson, coordinate_std, y[:, 2])

            idx_0_preds.append(pm25[:, 0].cpu().numpy())
            idx_0_targets.append(y[:, 2, 0].cpu().numpy())

            all_preds.append(pm25.cpu().numpy())
            all_targets.append(y[:, 2].squeeze(1).cpu().numpy())

    idx_preds = np.concatenate(idx_0_preds)
    idx_targets = np.concatenate(idx_0_targets)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # 反归一化PM2.5
    pm25_scaler = scalers['pm2_5']
    idx_preds = pm25_scaler.inverse_transform(idx_preds.reshape(-1, 1)).reshape(idx_preds.shape)
    idx_targets = pm25_scaler.inverse_transform(idx_targets.reshape(-1, 1)).reshape(idx_targets.shape)

    preds = pm25_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    targets = pm25_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)

    # 计算指标
    idx_mae = mean_absolute_error(idx_targets, idx_preds)
    idx_rmse = np.sqrt(mean_squared_error(idx_targets, idx_preds))
    idx_mape = mean_absolute_percentage_error(idx_targets, idx_preds)
    idx_R2 = r2_score_single(idx_targets, idx_preds)

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mape = mean_absolute_percentage_error(targets, preds)
    R2 = r2_score_single(targets, preds)

    logging.info(f'\nFinal Performance on Test Set:')
    logging.info(f'idx_0_MAE: {idx_mae:.2f}')
    logging.info(f'idx_0_RMSE: {idx_rmse:.2f}')
    logging.info(f'idx_0_MAPE: {idx_mape:.2f}')
    logging.info(f'idx_0_R2: {idx_R2:.2f}')

    logging.info(f'MAE: {mae:.2f}')
    logging.info(f'RMSE: {rmse:.2f}')
    logging.info(f'MAPE: {mape:.2f}')
    logging.info(f'R2: {R2:.2f}')

    # print(f'\nFinal Performance on Test Set:')
    # print(f'MAE: {mae:.2f}')
    # print(f'RMSE: {rmse:.2f}')

    plt.figure(figsize=(15, 10))

    # 1. PM2.5预测与真实值对比折线图
    plt.subplot(2, 1, 1)
    # 选择一个站点进行可视化
    station_index = 0
    # 只选取前24个时间步的数据
    num_steps = 24 * 3
    plt.plot(targets[:num_steps, station_index], label='True PM2.5', color='blue', linestyle='-', linewidth=2)
    plt.plot(preds[:num_steps, station_index], label='Predicted PM2.5', color='red', linestyle='--', linewidth=2)
    plt.title(f'True vs Predicted PM2.5 for Station {station_index} (First {num_steps} Time Steps)')
    plt.xlabel('Time Steps')
    plt.ylabel('PM2.5 Concentration (μg/m³)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    pic_path = os.path.join(save_dir, 'result.png')
    plt.savefig(pic_path, dpi=300)
    plt.show()

if __name__ == '__main__':

    torch.manual_seed(42)
    np.random.seed(42)

    data_tensor = np.load(cfg.dir_root + cfg.data_name)
    station_to_idx = load(cfg.dir_root + cfg.station_name)
    scalers = load(cfg.dir_root + cfg.scaler_name)

    station_coords = {}
    coordinate = []
    df = pd.read_csv('2024bj.csv')

    # 假设数据中经纬度的列名为'longitude'和'latitude'
    for station in station_to_idx:
        station_data = df[df['station_code'] == station].iloc[0]
        station_coords[station] = (station_data['longitude'], station_data['latitude'])
        coordinate.append([station_data['longitude'], station_data['latitude']])

    coordinate = torch.tensor(coordinate, dtype=torch.float32)
    coordinate_std = minmax_scale(coordinate).to(cfg.device)

    edge_index_distance = build_edge_index_by_distance(station_coords, cfg.threshold_km).to(cfg.device)
    edge_index_pearson = build_edge_index_by_pearson(data_tensor, cfg.threshold_pearson).to(cfg.device)

    graph_index_distance = edge2graph(edge_index_distance, is_self=True).to(cfg.device)
    graph_index_pearson = edge2graph(edge_index_pearson, is_self=True).to(cfg.device)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())

    save_name = f"{cfg.model_name}_{cfg.PRED_LEN}"
    save_dir = os.path.join(cfg.model_save_dir, save_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving path: {save_dir}")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'{save_name}_metrics.log'))
    console = logging.StreamHandler()  # Simultaneously output to console
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True



    save_name = f"{cfg.model_name}_{cfg.PRED_LEN}"
    pth = os.path.join(cfg.model_save_dir, save_name, f'{save_name}.pth')

    num_nodes = len(station_to_idx)


    if cfg.model_name == 'Hyper_reason':

        model = GCNLSTM(
            num_nodes=num_nodes,
            in_features=data_tensor.shape[-1],
            hidden_dim=cfg.HIDDEN_DIM,
            gcn_out=cfg.GCN_OUT,
            station_dim=cfg.staion_dim,
            time_dim=cfg.time_dim,
            hyper_dim=cfg.hyper_dim,
            curvature=cfg.curvature,
            tla_weight=cfg.tla_loss_weight,
            lstm_weight=cfg.lstm_loss_weight
        ).to(cfg.device)

    elif cfg.model_name == 'GC_LSTM':
        model = GCNLSTM(
            num_nodes=num_nodes,
            in_features=data_tensor.shape[-1],
            hidden_dim=cfg.HIDDEN_DIM,
            gcn_out=cfg.GCN_OUT,
            station_dim=cfg.staion_dim,
            time_dim=cfg.time_dim,
            hyper_dim=cfg.hyper_dim,
            curvature=cfg.curvature,
            tla_weight=cfg.tla_loss_weight,
            lstm_weight=cfg.lstm_loss_weight
        ).to(cfg.device)

    elif cfg.model_name == 'LSTM':
        model = LSTM(
            input_size=data_tensor.shape[-1],
            hidden_size=cfg.lstm_hidden
        ).to(cfg.device)

    elif cfg.model_name == 'Transformer':
        model = Trans(
            input_size=data_tensor.shape[-1],
            hidden_size=cfg.Transformer_hidden
        ).to(cfg.device)

    model_dict = torch.load(pth)
    model.load_state_dict(model_dict)

    metrics_gen()

    print('after')
