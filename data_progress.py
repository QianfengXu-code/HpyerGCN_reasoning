import pandas as pd
from tqdm import tqdm
from dataset import *
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    # 读取数据
    df = pd.read_csv(file_path)

    # 选择特征列 (排除无效列)
    feature_cols = ['aqi', 'so2', 'no2', 'co', 'o3', 'pm10', 'pm2_5',
                    'WIN_S_Avg_2mi', 'WIN_S_Avg_10mi', 'TEM', 'RHU', 'Alti']
    station_col = 'station_code'
    time_col = 'pubtime'

    # 处理缺失值
    df[feature_cols] = df[feature_cols].fillna(0)

    # 转换时间戳
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=[time_col, station_col])

    # 数据归一化
    scalers = {}
    for col in feature_cols:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        scalers[col] = scaler

    # 构建站点映射
    stations = sorted(df[station_col].unique())
    # stations.remove('3671A')
    station_to_idx = {s: i for i, s in enumerate(stations)}

    # 构建时间序列张量 [时间步, 站点数, 特征数]
    timesteps = sorted(df[time_col].unique())
    num_nodes = len(stations)
    num_features = len(feature_cols)

    data_tensor = np.zeros((len(timesteps), num_nodes, num_features))

    for t_idx, t in tqdm(enumerate(timesteps), desc="data_tensor_generating"):
        t_data = df[df[time_col] == t]
        for s in stations:
            s_data = t_data[t_data[station_col] == s]
            if not s_data.empty:
                s_idx = station_to_idx[s]
                data_tensor[t_idx, s_idx] = s_data[feature_cols].values[0]

    return data_tensor, station_to_idx, scalers

def load_and_preprocess_data_v2(file_path):
    # 读取数据
    df = pd.read_csv(file_path)

    # 选择特征列 (排除无效列)
    feature_cols = ['aqi', 'so2', 'no2', 'co', 'o3', 'pm10', 'pm2_5', 'pm2_5_24h',
                    'WIN_S_Avg_2mi', 'WIN_S_Avg_10mi', 'TEM', 'RHU', 'Alti', 'hour']
    feature_cols_remove = ['aqi', 'so2', 'no2', 'co', 'o3', 'pm10', 'pm2_5',
                    'WIN_S_Avg_2mi', 'WIN_S_Avg_10mi', 'TEM', 'RHU', 'Alti', 'hour']
    station_col = 'station_code'
    time_col = 'pubtime'

    # 确保 pubtime 是 datetime 类型
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour


    # 按站点分组处理
    station_dfs = {}
    for station_code, group in df.groupby(station_col):
        # 按时间排序
        group = group.sort_values(time_col)

        # 只处理 feature_cols 中的列
        features = group[feature_cols_remove]

        # 用前后数据的平均值填充缺失值（如果前后都有值）
        features_filled = features.interpolate(method='linear', limit_area='inside')

        # 对剩余缺失值用就近的非缺失值填充（向前或向后填充）
        features_filled = features_filled.ffill().bfill()

        # 将处理后的特征列放回原DataFrame
        group[feature_cols_remove] = features_filled

        # 存储处理后的站点数据
        station_dfs[station_code] = group

    # 构建站点映射
    stations = sorted(df[station_col].unique())

    # 填充缺失值
    for station_mark in stations:
        df_now = station_dfs[station_mark]
        rolling_mean = df_now['pm2_5'].rolling(24, min_periods=1, closed='left').mean().shift(1)
        rolling_mean.iloc[0] = rolling_mean.iloc[1]
        df_now['pm2_5_24h'] = df_now['pm2_5_24h'].fillna(rolling_mean).ffill().bfill()
        station_dfs[station_mark] = df_now

    df = pd.concat(station_dfs.values(), ignore_index=True)       # 忽略原始索引，生成新索引

    # 转换时间戳
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=[time_col, station_col])

    # 数据归一化
    scalers = {}
    for col in feature_cols:
        if col != 'hour':
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            scalers[col] = scaler


    stations.remove('3671A')
    station_to_idx = {s: i for i, s in enumerate(stations)}

    # 构建时间序列张量 [时间步, 站点数, 特征数]
    timesteps = sorted(df[time_col].unique())
    num_nodes = len(stations)
    num_features = len(feature_cols)

    data_tensor = np.zeros((len(timesteps), num_nodes, num_features))

    for t_idx, t in tqdm(enumerate(timesteps), desc="data_tensor_generating"):
        t_data = df[df[time_col] == t]
        for s in stations:
            s_data = t_data[t_data[station_col] == s]
            if not s_data.empty:
                s_idx = station_to_idx[s]
                data_tensor[t_idx, s_idx] = s_data[feature_cols].values[0]

    return data_tensor, station_to_idx, scalers

