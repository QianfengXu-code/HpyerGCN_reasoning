import math
import time
import copy  # 用于模型深拷贝
import zipfile
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from model_test import model_save
import config
import config as cfg
from data_progress import *
from models import GCNLSTM, LSTM, Trans, nn, MultiTaskLoss
from tqdm import tqdm
import os
import logging
from joblib import dump, load
from pathlib import Path
from utils import *
import torch.nn.functional as F
from torch.utils.data import DataLoader



# --------------------
# 主程序流程
# --------------------
if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)


    # 记录开始时间
    start_time = time.time()

    # 1. 加载并预处理数据
    if cfg.need_preprocess:
        data_tensor, station_to_idx, scalers = load_and_preprocess_data_v2('2024bj.csv')

        np.save('./data2/data_tensor.npy', data_tensor)
        dump(station_to_idx, './data2/station_to_idx.joblib')
        dump(scalers, './data2/scalers.joblib')

    data_tensor = np.load(cfg.dir_root + cfg.data_name)
    station_to_idx = load(cfg.dir_root + cfg.station_name)
    scalers = load(cfg.dir_root + cfg.scaler_name)

    # 重新读取数据以获取经纬度信息
    df = pd.read_csv('2024bj.csv')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())

    save_name = f"{config.model_name}_{config.PRED_LEN}"
    save_dir = os.path.join(config.model_save_dir, save_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving path: {save_dir}")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'{save_name}.log'))
    console = logging.StreamHandler()  # Simultaneously output to console
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    zipf = zipfile.ZipFile(file=os.path.join(save_dir, 'codes.zip'), mode='a', compression=zipfile.ZIP_DEFLATED)
    zipdir(Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # 2. 构建图结构（使用实际经纬度数据）
    station_coords = {}
    coordinate = []
    # 假设数据中经纬度的列名为'longitude'和'latitude'
    for station in station_to_idx:
        station_data = df[df['station_code'] == station].iloc[0]
        station_coords[station] = (station_data['longitude'], station_data['latitude'])
        coordinate.append([station_data['longitude'], station_data['latitude']])

    coordinate = torch.tensor(coordinate, dtype=torch.float32)
    coordinate_std = minmax_scale(coordinate).to(cfg.device)

    # edge_index基于地理距离（10km阈值）
    edge_index_distance = build_edge_index_by_distance(station_coords, cfg.threshold_km).to(cfg.device)
    edge_index_pearson = build_edge_index_by_pearson(data_tensor, cfg.threshold_pearson).to(cfg.device)

    graph_index_distance = edge2graph(edge_index_distance, is_self=True).to(cfg.device)
    graph_index_pearson = edge2graph(edge_index_pearson, is_self=True).to(cfg.device)

    # 3. 创建数据集并划分
    # dataset = AirQualityDataset(data_tensor, cfg.SEQ_LEN, cfg.PRED_LEN)
    # total_size = len(dataset)
    # train_size = int(0.8 * total_size)  # 60% 训练
    # val_size = int(0.1 * total_size)  # 20% 验证
    # test_size = total_size - train_size - val_size  # 20% 测试
    #
    # # train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    # # val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    # # test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
    #
    #
    #
    # # 使用随机划分
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, val_size, test_size])
    #
    # # 创建数据加载器
    # train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    train_loader, val_loader, test_loader = dataloader_build(data_tensor, cfg)

    # 4. 初始化模型
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

    # 5. 训练配置
    optimizer = torch.optim.Adam(params=(list(model.parameters())), lr=cfg.LR)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 每 5 epoch 学习率 ×0.1
    criterion = nn.MSELoss()

    # 记录损失
    train_losses = []
    val_losses = []
    test_losses = []


    # 早停机制变量
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    early_stop = False

    # 在日志中写入参数设置情况
    file_path = "config.py"  # 替换为你的文件路径
    variables = extract_vars_dynamic(file_path)
    for name, value in variables.items():
        logging.info(f"{name}: {repr(value)}")

    # torch.autograd.set_detect_anomaly(True)

    # 6. 训练循环
    for epoch in range(cfg.EPOCHS):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        model.train()
        total_train_loss = 0

        # 训练阶段
        for idx, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(cfg.device), y.to(cfg.device)

            optimizer.zero_grad()
            if cfg.model_name == 'LSTM' or cfg.model_name == 'Transformer':
                pm25 = model(x)
                loss = criterion(pm25, y[:, 2].squeeze(1))
            else:
                so2, pm10, pm25, loss_tla = model(x, edge_index_distance, edge_index_pearson, graph_index_distance,
                                                  graph_index_pearson, coordinate_std, y[:, 2])

                loss_so2 = criterion(so2, y[:, 0].squeeze(1))
                loss_pm10 = criterion(pm10, y[:, 1].squeeze(1))
                loss_pm25 = criterion(pm25, y[:, 2].squeeze(1))

                w = torch.sigmoid(model.loss_weight)

                loss = w * loss_pm25 + loss_tla
                # loss = loss_pm25

            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN梯度出现在: {name}")
                    print(f"gcn1.bias grad: max={model.gcn1.bias.grad.max()}, min={model.gcn1.bias.grad.min()}")
                    if torch.isnan(param.data).any():
                        print(f"NaN参数出现在: {name}")
                    break

            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)

                if cfg.model_name == 'LSTM' or cfg.model_name == 'Transformer':

                    pm25 = model(x)

                else:

                    so2, pm10, pm25, loss_tla = model(x, edge_index_distance, edge_index_pearson, graph_index_distance,
                                                      graph_index_pearson, coordinate_std, y[:, 2])
                total_val_loss += criterion(pm25, y[:, 2].squeeze(1)).item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # 测试阶段（仅用于监控，不用于早停决策）
        total_test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)

                if cfg.model_name == 'LSTM' or cfg.model_name == 'Transformer':
                    pm25 = model(x)

                else:

                    so2, pm10, pm25, loss_tla = model(x, edge_index_distance, edge_index_pearson, graph_index_distance,
                                                      graph_index_pearson, coordinate_std, y[:, 2])
                total_test_loss += criterion(pm25, y[:, 2].squeeze(1)).item()

        test_loss = total_test_loss / len(test_loader)
        test_losses.append(test_loss)

        # 打印所有损失
        logging.info(f'Epoch {epoch + 1}/{cfg.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}')


        # 早停机制检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"Validation loss improved to {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve}/{cfg.PATIENCE} epochs")

            if epochs_no_improve >= cfg.PATIENCE:
                early_stop = True

        if test_loss < best_test_loss:
            best_model = copy.deepcopy(model.state_dict())  # 保存最佳模型

        scheduler.step()


    # 计算训练时间
    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time // 60:.0f}m {training_time % 60:.0f}s")

    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
        print("Loaded best model based on validation loss")

    if config.need_save:

        pth = os.path.join(config.model_save_dir, save_name, f'{save_name}.pth')
        model_save(model, pth)
        print(f"successfully saved @ {pth}")

    print("program has already finished")

    # 7. 在测试集上评估最佳模型

