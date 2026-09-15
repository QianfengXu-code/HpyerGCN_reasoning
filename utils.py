import os
import numpy as np
import torch
import importlib.util
import sys
from scipy import stats
from config import *
import matplotlib.pyplot as plt
from dataset import AirQualityDataset
from torch.utils.data import DataLoader



def build_edge_index_by_distance(station_coords, threshold_km=10):
    """
    基于经纬度和距离阈值构建edge_index
    """
    stations = list(station_coords.keys())
    num_nodes = len(stations)
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                edge_list.append([i, j])  # 自连接
            else:
                lon1, lat1 = station_coords[stations[i]]
                lon2, lat2 = station_coords[stations[j]]
                dist = haversine(lon1, lat1, lon2, lat2)
                if dist <= threshold_km:
                    edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def build_edge_index_by_pearson(data_tensor, threshold_pearson=10):
    """
    基于pearson构建edge_index
    """
    station_num = data_tensor.shape[1]
    stations = {}
    edge_list = []

    for station in range(station_num):
        stations[station] = data_tensor[:, station, 6]

    for station_first in range(station_num):
        v_list_1 = stations[station_first]
        for station_second in range(station_first, station_num):
            v_list_2 = stations[station_second]
            r, p = stats.pearsonr(v_list_1, v_list_2)
            if r > threshold_pearson:
                edge_list.append([station_first, station_second])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index


def haversine(lon1, lat1, lon2, lat2):
    """
    计算两点之间的球面距离（单位：千米）
    """
    R = 6371  # 地球半径，单位为千米
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def edge2graph(edge, is_self=False):
    graph = np.zeros([23, 23])
    for i, j in edge.t():
        graph[i, j] = 1
        graph[j, i] = 1
    if not is_self:
        for idx in range(24):
            graph[idx, idx] = 0
    graph_tensor = torch.tensor(graph, dtype=torch.float32)
    return graph_tensor


def minmax_scale(coords):
    lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
    lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
    coords[:, 0] = (coords[:, 0] - lat_min) / (lat_max - lat_min)
    coords[:, 1] = (coords[:, 1] - lon_min) / (lon_max - lon_min)
    return coords


def zipdir(path, zipf, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                # os.path.splitext 分开文件名和扩展名 此处要提取出.py
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                zipf.write(filename, arcname)


def lorentz_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """修正版：Minkowski内积（符号差 (-++...+)）"""
    return -a[..., 0] * b[..., 0] + torch.sum(a[..., 1:] * b[..., 1:], dim=-1)


def lorentz_norm(v: torch.Tensor) -> torch.Tensor:
    """修正版：切空间向量的洛伦兹范数（内积正定）"""
    inner = lorentz_inner_product(v, v)
    return torch.sqrt(torch.clamp(inner, min=1e-10))  # 此时inner恒正，可安全开方


def get_pole(d, c) -> torch.Tensor:
    pole_np = np.zeros(d)
    pole = torch.tensor(pole_np, dtype=torch.float32, device=device)
    pole[0] = torch.sqrt(c)
    return pole


def gen_batch_gragh(i_indices, j_indices, batch_size, nodes):
    neighbor_nums = i_indices.shape[-1]

    list_i = torch.cat([i_indices] * batch_size, dim=-1)
    list_j = torch.cat([j_indices] * batch_size, dim=-1)

    mask = []
    mask_ones = np.ones(neighbor_nums)

    for batch_idx in range(batch_size):
        mask_batch = torch.tensor(batch_idx * nodes * mask_ones, dtype=torch.int64)
        mask.append(mask_batch)

    mask_tensor = torch.cat(mask, dim=-1).to(device)
    list_batch_i = list_i + mask_tensor
    list_batch_j = list_j + mask_tensor

    return list_batch_i, list_batch_j


def extract_vars_dynamic(file_path):
    """动态执行文件并提取全局变量"""
    # 构造唯一的模块名
    module_name = file_path.replace('.py', '').replace('/', '.').replace('\\', '.')

    # 动态加载模块
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # 执行文件代码

    # 提取变量（过滤内置变量）
    variables = {
        k: v for k, v in vars(module).items()
        if not k.startswith('__') and not k.endswith('__')
    }
    return variables


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    """
      计算 MAPE（平均绝对百分比误差）

      参数:
          y_true: 真实值数组
          y_pred: 预测值数组
          epsilon: 防止除以零的小常数（默认 1e-10）

      返回:
          MAPE（百分比形式，例如 5.0 表示 5%）
      """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.clip(y_pred, 1, 1000)
    y_true = np.clip(y_true, 1, 1000)
    ape = np.abs((y_true - y_pred) / y_true)  # 避免除以零
    mean_ape = np.mean(ape)

    return mean_ape * 100  # 转换为百分比


def r2_score_single(O, P):
    """
    计算整体 R²（无论输入形状如何，均返回单个标量）。
    将观测值和预测值展平为 1D 数组后计算 R²。

    参数:
        O (array-like): 观测值（实际值），任意形状（如 [n]、[n, b]、[n, h, w] 等）
        P (array-like): 预测值，形状需与 O 完全一致

    返回:
        float: 整体 R² 值（标量）
    """
    # 转换为 numpy 数组并展平为 1D
    O = np.asarray(O).flatten()
    P = np.asarray(P).flatten()

    # 检查展平后长度是否一致
    if len(O) != len(P):
        raise ValueError(f"展平后观测值和预测值长度不一致：{len(O)} vs {len(P)}")

    # 计算总平方和 SS_tot（观测值与均值的平方差之和）
    O_mean = np.mean(O)
    ss_tot = np.sum((O - O_mean) ** 2)

    # 计算残差平方和 SS_res（观测值与预测值的平方差之和）
    ss_res = np.sum((O - P) ** 2)

    # 处理 SS_tot=0 的情况（观测值无波动）
    if ss_tot == 0:
        return 1.0  # 若观测值全相同，R² 定义为 1

    # 计算整体 R²
    r2 = 1 - (ss_res / ss_tot)
    return r2


def dataloader_build(data_tensor, cfg):

    dataset = AirQualityDataset(data_tensor, cfg.SEQ_LEN, cfg.PRED_LEN)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 60% 训练
    val_size = int(0.1 * total_size)  # 20% 验证
    test_size = total_size - train_size - val_size  # 20% 测试

    # train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    # val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    # test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))

    # 使用随机划分
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

