import numpy as np
import torch
from tla_loss import *
from utils import *
import torch.nn as nn
import math
import torch.nn.functional as F
import config as cfg
from hyper import HGCN
from utils import gen_batch_gragh
from torch_geometric.nn import GCNConv
from manual_LSTM import CustomLSTM, CustomLSTMCell

class GCNLSTM(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, gcn_out, station_dim,
                 time_dim, hyper_dim, lstm_layers=1, curvature=1,
                 tla_weight=-2.2, lstm_weight=2.2):
        super(GCNLSTM, self).__init__()
        self.num_nodes = num_nodes
        self.nodes_idx = torch.tensor(np.arange(self.num_nodes), dtype=torch.int64, device=device)
        self.station_emb = nn.Embedding(num_nodes, station_dim)
        self.time_emb = nn.Embedding(24, time_dim)

        self.hyper_in_dim = 2

        self.gcn1 = GCNConv(in_features - 1 + time_dim, gcn_out)
        self.gcn2 = GCNConv(gcn_out, gcn_out)

        self.gcn3 = GCNConv(in_features - 1 + time_dim, gcn_out)
        self.gcn4 = GCNConv(gcn_out, gcn_out)

        self.Hyper_GCN_dist = HGCN(self.hyper_in_dim, hyper_dim, 4, curvature,
                                  ctype='sid', depth=3, device=cfg.device)

        self.Hyper_GCN_pear = HGCN(self.hyper_in_dim, hyper_dim, 4, curvature,
                                  ctype='sid', depth=3, device=cfg.device)

        self.silu = BSiLU()

        self.fusion_1 = nn.Linear(3 * gcn_out, gcn_out)

        self.att = SelfAttention(hidden_dim, 4, num_nodes)

        # LSTM输入: [batch, seq_len, gcn_out]
        # self.lstm = nn.LSTM(2 * gcn_out + 12 + staion_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)

        if cfg.need_hyper and cfg.model_name == 'Hyper_reason':

            self.manual_lstm = CustomLSTM(2 * gcn_out + 2 * hyper_dim + staion_dim, hidden_dim, num_layers=lstm_layers)

        else:

            self.manual_lstm = CustomLSTM(2 * gcn_out, hidden_dim, num_layers=lstm_layers)

        self.tla_weight = torch.sigmoid(torch.tensor([tla_weight], dtype=torch.float32, device=device))
        # self.sta_weight = nn.Parameter(torch.tensor([0], dtype=torch.float32, device=device))
        self.loss_weight = torch.sigmoid(torch.tensor([lstm_weight], dtype=torch.float32, device=device))


        self.fc_so2 = nn.Linear(hidden_dim, 1)  # 预测PM2.5
        self.fc_pm10 = nn.Linear(hidden_dim, 1)  # 预测PM2.5
        self.fc_pm25 = nn.Linear(hidden_dim, 1)  # 预测PM2.5

    def forward(self, x_double, edge_index_distance, edge_index_pearson,
                graph_index_distance, graph_index_pearson, coordinate_std, y):

        if cfg.model_name == 'GC_LSTM':

            # x: [batch, seq_len, nodes, features]
            x = x_double.float()
            batch_size, seq_len, num_nodes, _ = x.size()
            pm25_feature = x[:, :, :, 6].permute(0, 2, 1).reshape(-1, seq_len)


            # 从两个临近图，对坐标进行双曲图卷积
            # 考虑修改之后，再对其他特征卷积?

            # hyper_dist = self.Hyper_GCN_dist(coordinate_std, graph_index_distance)
            # hyper_pear = self.Hyper_GCN_pear(coordinate_std, graph_index_pearson)
            #
            # hyper_dist_batch = hyper_dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
            # hyper_pear_batch = hyper_pear.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)

            # distance_mask = graph_index_distance.bool()
            # pearson_mask = graph_index_pearson.bool()

            # upper_tri_distance_mask = torch.triu(distance_mask, diagonal=1)
            # upper_tri_pearson_mask = torch.triu(pearson_mask, diagonal=1)

            # 两个图的 目标节点与之邻接节点 [M] M表示邻接总数
            # 每个batch的邻接情况是相同的，只需要改变索引值就可以了(相差24 * batch的倍数)
            # 即，batch个图的邻接节点总数为 [batch * M]

            # i_distance_indices, j_distance_indices = torch.where(upper_tri_distance_mask)
            # i_pearson_indices, j_pearson_indices = torch.where(upper_tri_pearson_mask)

            # emb_weight = self.station_emb.weight
            # hyper_in = torch.cat([coordinate_std, emb_weight], dim=-1)
            # hyper_in = coordinate_std

            # hyper_dist = self.Hyper_GCN_dist(hyper_in, i_distance_indices, j_distance_indices)
            # hyper_pear = self.Hyper_GCN_pear(hyper_in, i_pearson_indices, j_pearson_indices)

            # hyper_dist_batch = hyper_dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
            # hyper_pear_batch = hyper_pear.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)

            # 构造邻接大图[batch * nodes, batch * nodes]

            # i_distance_batch, j_distance_batch = gen_batch_gragh(i_distance_indices, j_distance_indices,
            #                                                      batch_size, num_nodes)
            #
            # i_pearson_batch, j_pearson_batch = gen_batch_gragh(i_pearson_indices, j_pearson_indices,
            #                                                      batch_size, num_nodes)

            # 处理每一个时间步
            # gcn_outputs = []
            # for t in range(seq_len):
            #     # 展平节点维度用于GCN
            #     x_t = x[:, t].reshape(-1, x.size(-1))  # [batch*nodes, features]
            #     # 双曲图卷积操作-dist
            #     dist_emb = self.Hyper_GCN_dist(x_t, i_distance_batch, j_distance_batch)
            #     # 双曲图卷积操作-pear
            #     pear_emb = self.Hyper_GCN_pear(x_t, i_pearson_batch, j_pearson_batch)
            #
            #     hyper_gcn_emb = torch.cat([dist_emb, pear_emb], dim=-1)
            #
            #     gcn_outputs.append(hyper_gcn_emb.view(batch_size, num_nodes, -1))

            # 处理每个时间步
            gcn_outputs = []
            for t in range(seq_len):
                # 展平节点维度用于GCN
                x_t = x[:, t].reshape(-1, x.size(-1))  # [batch*nodes, features]
                x_time = x_t[:, -1]
                time_x = self.time_emb(x_time.to(torch.int))
                x_t = torch.cat([x_t[:, :-1], time_x], dim=-1)

                # GCN处理
                h = self.gcn1(x_t, edge_index_distance)
                gcn_distance = self.silu(self.gcn2(h, edge_index_distance))

                h = self.gcn3(x_t, edge_index_pearson)
                gcn_pearson = self.silu(self.gcn4(h, edge_index_pearson))

                gcn_emb = torch.cat([gcn_distance, gcn_pearson], dim=-1)

                gcn_outputs.append(gcn_emb.view(batch_size, num_nodes, -1))

            # 组合时间序列 [batch, seq_len, nodes, gcn_out]
            gcn_sta = torch.stack(gcn_outputs, dim=1)
            # gcn_hyper = torch.cat([gcn_sta, hyper_dist_batch, hyper_pear_batch], dim=-1)
            gcn_hyper = gcn_sta

            batch, seq, nodes, dim = gcn_hyper.shape

            # 准备LSTM输入 [batch*nodes, seq_len, gcn_out]
            gcn_seq = gcn_hyper.permute(0, 2, 1, 3).reshape(-1, seq_len, dim)
            lstm_in = gcn_seq

            # nodes_emb = self.station_emb(self.nodes_idx)
            # batch_emb = nodes_emb.repeat(batch_size, 1).unsqueeze(1).repeat(1, seq_len, 1)
            #
            # lstm_in = torch.cat([gcn_seq, batch_emb], dim=-1)

            # LSTM处理
            # lstm_out, _ = self.lstm(gcn_seq)  # [batch*nodes, seq_len, hidden_dim]
            lstm_out, _ = self.manual_lstm(lstm_in, pm25_feature)  # [batch*nodes, seq_len, hidden_dim]

            # b = random.randint(0, seq_len - 1 - 1)  # 公式(13)：均匀采样 b ~ Unif {1, ..., k-1}
            # Bk = lstm_out[:, -1]
            # Bb = lstm_out[:, b]
            #
            # loss_tla = step_level_alignment_loss(Bk, Bb, seq_len)
            # loss_sta = trajectory_level_alignment_loss(lstm_out[:, -1], y)
            #
            # loss_ex = self.tla_weight * loss_tla + (1 - self.tla_weight) * loss_sta
            loss_ex = 0

            # 取最后一个时间步输出
            last_out = lstm_out[:, -1]  # [batch*nodes, hidden_dim]
            # value = self.att(last_out.view(batch_size, num_nodes, -1))
            # value = value.view(batch_size * num_nodes, -1)
            value = last_out

            # 预测PM2.5
            pred_so2 = self.fc_so2(value)  # [batch*nodes, 1]
            pred_pm10 = self.fc_pm10(value)  # [batch*nodes, 1]
            pred_pm25 = self.fc_pm25(value)  # [batch*nodes, 1]

            so2 = pred_so2.view(batch_size, num_nodes)  # [batch, nodes]
            pm10 = pred_pm10.view(batch_size, num_nodes)  # [batch, nodes]
            pm25 = pred_pm25.view(batch_size, num_nodes)  # [batch, nodes]

        elif not cfg.need_hyper:

            # x: [batch, seq_len, nodes, features]
            x = x_double.float()
            batch_size, seq_len, num_nodes, _ = x.size()
            pm25_feature = x[:, :, :, 6].permute(0, 2, 1).reshape(-1, seq_len)

            # 从两个临近图，对坐标进行双曲图卷积
            # 考虑修改之后，再对其他特征卷积?

            # hyper_dist = self.Hyper_GCN_dist(coordinate_std, graph_index_distance)
            # hyper_pear = self.Hyper_GCN_pear(coordinate_std, graph_index_pearson)
            #
            # hyper_dist_batch = hyper_dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
            # hyper_pear_batch = hyper_pear.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)

            # distance_mask = graph_index_distance.bool()
            # pearson_mask = graph_index_pearson.bool()

            # upper_tri_distance_mask = torch.triu(distance_mask, diagonal=1)
            # upper_tri_pearson_mask = torch.triu(pearson_mask, diagonal=1)

            # 两个图的 目标节点与之邻接节点 [M] M表示邻接总数
            # 每个batch的邻接情况是相同的，只需要改变索引值就可以了(相差24 * batch的倍数)
            # 即，batch个图的邻接节点总数为 [batch * M]

            # i_distance_indices, j_distance_indices = torch.where(upper_tri_distance_mask)
            # i_pearson_indices, j_pearson_indices = torch.where(upper_tri_pearson_mask)

            # emb_weight = self.station_emb.weight
            # hyper_in = torch.cat([coordinate_std, emb_weight], dim=-1)
            # hyper_in = coordinate_std

            # hyper_dist = self.Hyper_GCN_dist(hyper_in, i_distance_indices, j_distance_indices)
            # hyper_pear = self.Hyper_GCN_pear(hyper_in, i_pearson_indices, j_pearson_indices)

            # hyper_dist_batch = hyper_dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
            # hyper_pear_batch = hyper_pear.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)

            # 构造邻接大图[batch * nodes, batch * nodes]

            # i_distance_batch, j_distance_batch = gen_batch_gragh(i_distance_indices, j_distance_indices,
            #                                                      batch_size, num_nodes)
            #
            # i_pearson_batch, j_pearson_batch = gen_batch_gragh(i_pearson_indices, j_pearson_indices,
            #                                                      batch_size, num_nodes)

            # 处理每一个时间步
            # gcn_outputs = []
            # for t in range(seq_len):
            #     # 展平节点维度用于GCN
            #     x_t = x[:, t].reshape(-1, x.size(-1))  # [batch*nodes, features]
            #     # 双曲图卷积操作-dist
            #     dist_emb = self.Hyper_GCN_dist(x_t, i_distance_batch, j_distance_batch)
            #     # 双曲图卷积操作-pear
            #     pear_emb = self.Hyper_GCN_pear(x_t, i_pearson_batch, j_pearson_batch)
            #
            #     hyper_gcn_emb = torch.cat([dist_emb, pear_emb], dim=-1)
            #
            #     gcn_outputs.append(hyper_gcn_emb.view(batch_size, num_nodes, -1))

            # 处理每个时间步
            gcn_outputs = []
            for t in range(seq_len):
                # 展平节点维度用于GCN
                x_t = x[:, t].reshape(-1, x.size(-1))  # [batch*nodes, features]
                x_time = x_t[:, -1]
                time_x = self.time_emb(x_time.to(torch.int))
                x_t = torch.cat([x_t[:, :-1], time_x], dim=-1)

                # GCN处理
                h = self.gcn1(x_t, edge_index_distance)
                gcn_distance = self.silu(self.gcn2(h, edge_index_distance))

                h = self.gcn3(x_t, edge_index_pearson)
                gcn_pearson = self.silu(self.gcn4(h, edge_index_pearson))

                gcn_emb = torch.cat([gcn_distance, gcn_pearson], dim=-1)

                gcn_outputs.append(gcn_emb.view(batch_size, num_nodes, -1))

            # 组合时间序列 [batch, seq_len, nodes, gcn_out]
            gcn_sta = torch.stack(gcn_outputs, dim=1)
            # gcn_hyper = torch.cat([gcn_sta, hyper_dist_batch, hyper_pear_batch], dim=-1)
            gcn_hyper = gcn_sta

            batch, seq, nodes, dim = gcn_hyper.shape

            # 准备LSTM输入 [batch*nodes, seq_len, gcn_out]
            gcn_seq = gcn_hyper.permute(0, 2, 1, 3).reshape(-1, seq_len, dim)

            nodes_emb = self.station_emb(self.nodes_idx)
            batch_emb = nodes_emb.repeat(batch_size, 1).unsqueeze(1).repeat(1, seq_len, 1)

            lstm_in = torch.cat([gcn_seq, batch_emb], dim=-1)

            # LSTM处理
            # lstm_out, _ = self.lstm(gcn_seq)  # [batch*nodes, seq_len, hidden_dim]
            lstm_out, _ = self.manual_lstm(lstm_in, pm25_feature)  # [batch*nodes, seq_len, hidden_dim]

            b = random.randint(0, seq_len - 1 - 1)  # 公式(13)：均匀采样 b ~ Unif {1, ..., k-1}
            Bk = lstm_out[:, -1]
            Bb = lstm_out[:, b]

            loss_tla = step_level_alignment_loss(Bk, Bb, seq_len)
            loss_sta = trajectory_level_alignment_loss(lstm_out[:, -1], y)

            loss_ex = self.tla_weight * loss_tla + (1 - self.tla_weight) * loss_sta

            # 取最后一个时间步输出
            last_out = lstm_out[:, -1]  # [batch*nodes, hidden_dim]
            value = self.att(last_out.view(batch_size, num_nodes, -1))
            value = value.view(batch_size * num_nodes, -1)

            # 预测PM2.5
            pred_so2 = self.fc_so2(value)  # [batch*nodes, 1]
            pred_pm10 = self.fc_pm10(value)  # [batch*nodes, 1]
            pred_pm25 = self.fc_pm25(value)  # [batch*nodes, 1]

            so2 = pred_so2.view(batch_size, num_nodes)  # [batch, nodes]
            pm10 = pred_pm10.view(batch_size, num_nodes)  # [batch, nodes]
            pm25 = pred_pm25.view(batch_size, num_nodes)  # [batch, nodes]

        else:

            # x: [batch, seq_len, nodes, features]
            x = x_double.float()
            batch_size, seq_len, num_nodes, _ = x.size()
            pm25_feature = x[:, :, :, 6].permute(0, 2, 1).reshape(-1, seq_len)
            batch_size, seq_len, num_nodes, _ = x.size()

            # 从两个临近图，对坐标进行双曲图卷积
            # 考虑修改之后，再对其他特征卷积?

            # hyper_dist = self.Hyper_GCN_dist(coordinate_std, graph_index_distance)
            # hyper_pear = self.Hyper_GCN_pear(coordinate_std, graph_index_pearson)
            #
            # hyper_dist_batch = hyper_dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
            # hyper_pear_batch = hyper_pear.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)

            distance_mask = graph_index_distance.bool()
            pearson_mask = graph_index_pearson.bool()

            upper_tri_distance_mask = torch.triu(distance_mask, diagonal=1)
            upper_tri_pearson_mask = torch.triu(pearson_mask, diagonal=1)

            # 两个图的 目标节点与之邻接节点 [M] M表示邻接总数
            # 每个batch的邻接情况是相同的，只需要改变索引值就可以了(相差24 * batch的倍数)
            # 即，batch个图的邻接节点总数为 [batch * M]
            i_distance_indices, j_distance_indices = torch.where(upper_tri_distance_mask)
            i_pearson_indices, j_pearson_indices = torch.where(upper_tri_pearson_mask)

            # emb_weight = self.station_emb.weight
            # hyper_in = torch.cat([coordinate_std, emb_weight], dim=-1)
            hyper_in = coordinate_std

            hyper_dist = self.Hyper_GCN_dist(hyper_in, i_distance_indices, j_distance_indices)
            hyper_pear = self.Hyper_GCN_pear(hyper_in, i_pearson_indices, j_pearson_indices)

            hyper_dist_batch = hyper_dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
            hyper_pear_batch = hyper_pear.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)

            # 构造邻接大图[batch * nodes, batch * nodes]

            # i_distance_batch, j_distance_batch = gen_batch_gragh(i_distance_indices, j_distance_indices,
            #                                                      batch_size, num_nodes)
            #
            # i_pearson_batch, j_pearson_batch = gen_batch_gragh(i_pearson_indices, j_pearson_indices,
            #                                                      batch_size, num_nodes)

            # 处理每一个时间步
            # gcn_outputs = []
            # for t in range(seq_len):
            #     # 展平节点维度用于GCN
            #     x_t = x[:, t].reshape(-1, x.size(-1))  # [batch*nodes, features]
            #     # 双曲图卷积操作-dist
            #     dist_emb = self.Hyper_GCN_dist(x_t, i_distance_batch, j_distance_batch)
            #     # 双曲图卷积操作-pear
            #     pear_emb = self.Hyper_GCN_pear(x_t, i_pearson_batch, j_pearson_batch)
            #
            #     hyper_gcn_emb = torch.cat([dist_emb, pear_emb], dim=-1)
            #
            #     gcn_outputs.append(hyper_gcn_emb.view(batch_size, num_nodes, -1))

            # 处理每个时间步
            gcn_outputs = []
            for t in range(seq_len):
                # 展平节点维度用于GCN
                x_t = x[:, t].reshape(-1, x.size(-1))  # [batch*nodes, features]
                x_time = x_t[:, -1]
                time_x = self.time_emb(x_time.to(torch.int))
                x_t = torch.cat([x_t[:, :-1], time_x], dim=-1)

                # GCN处理
                h = self.gcn1(x_t, edge_index_distance)
                gcn_distance = self.silu(self.gcn2(h, edge_index_distance))

                h = self.gcn3(x_t, edge_index_pearson)
                gcn_pearson = self.silu(self.gcn4(h, edge_index_pearson))

                gcn_emb = torch.cat([gcn_distance, gcn_pearson],  dim=-1)

                gcn_outputs.append(gcn_emb.view(batch_size, num_nodes, -1))

            # 组合时间序列 [batch, seq_len, nodes, gcn_out]
            gcn_sta = torch.stack(gcn_outputs, dim=1)
            gcn_hyper = torch.cat([gcn_sta, hyper_dist_batch, hyper_pear_batch], dim=-1)
            # gcn_hyper = gcn_sta

            batch, seq, nodes, dim = gcn_hyper.shape

            # 准备LSTM输入 [batch*nodes, seq_len, gcn_out]
            gcn_seq = gcn_hyper.permute(0, 2, 1, 3).reshape(-1, seq_len, dim)

            nodes_emb = self.station_emb(self.nodes_idx)
            batch_emb = nodes_emb.repeat(batch_size, 1).unsqueeze(1).repeat(1, seq_len, 1)

            lstm_in = torch.cat([gcn_seq, batch_emb], dim=-1)

            # LSTM处理
            # lstm_out, _ = self.lstm(gcn_seq)  # [batch*nodes, seq_len, hidden_dim]
            lstm_out, _ = self.manual_lstm(lstm_in, pm25_feature)  # [batch*nodes, seq_len, hidden_dim]

            # b = random.randint(0, seq_len - 1 - 1)  # 公式(13)：均匀采样 b ~ Unif {1, ..., k-1}
            # Bk = lstm_out[:, -1]
            # Bb = lstm_out[:, b]

            # loss_tla = step_level_alignment_loss(Bk, Bb, seq_len)
            # loss_sta = trajectory_level_alignment_loss(lstm_out[:, -1], y)
            #
            # loss_ex = loss_sta

            if cfg.need_reasoning:
                last_out = lstm_out[:, -1]  # [batch*nodes, hidden_dim]

                value = self.att(last_out.view(batch_size, num_nodes, -1))
                value = value.view(batch_size * num_nodes, -1)

                b = random.randint(0, seq_len - 1 - 1)  # 公式(13)：均匀采样 b ~ Unif {1, ..., k-1}
                Bk = lstm_out[:, -1]
                Bb = lstm_out[:, b]

                loss_tla = step_level_alignment_loss(Bk, Bb, seq_len)
                loss_sta = trajectory_level_alignment_loss(lstm_out[:, -1], y)

                loss_ex = loss_sta + 0.05 * loss_tla

            else:
                value = lstm_out[:, -1]
                value = value.view(batch_size * num_nodes, -1)
                loss_ex = 0

            # 预测PM2.5
            pred_so2 = self.fc_so2(value)  # [batch*nodes, 1]
            pred_pm10 = self.fc_pm10(value)  # [batch*nodes, 1]
            pred_pm25 = self.fc_pm25(value)  # [batch*nodes, 1]

            so2 = pred_so2.view(batch_size, num_nodes)  # [batch, nodes]
            pm10 = pred_pm10.view(batch_size, num_nodes)  # [batch, nodes]
            pm25 = pred_pm25.view(batch_size, num_nodes)  # [batch, nodes]

        return so2, pm10, pm25, loss_ex


class Trans(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Trans, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.att = SelfAttention(hidden_size, 1, 23)

        self.fc_pm25 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, nodes, features]
        batch_size, seq_len, num_nodes, num_features = x.shape

        # 合并 batch 和 nodes 维度 [batch * nodes, seq_len, features]
        x = x.transpose(1, 2)  # [batch, nodes, seq_len, features]
        x = x.reshape(batch_size * num_nodes, seq_len, num_features)

        # LSTM 处理 [batch * nodes, seq_len, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 获取最后一个时间步的输出 [batch * nodes, hidden_size]
        # 也可以用 h_n.squeeze(0) 来获取隐状态
        output = lstm_out[:, -1, :]
        value = self.att(output.view(batch_size, num_nodes, -1))
        pm25 = self.fc_pm25(value)
        pm25_out = pm25.view(batch_size, num_nodes)

        return pm25_out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.fc_pm25 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, nodes, features]
        batch_size, seq_len, num_nodes, num_features = x.shape

        # 合并 batch 和 nodes 维度 [batch * nodes, seq_len, features]
        x = x.transpose(1, 2)  # [batch, nodes, seq_len, features]
        x = x.reshape(batch_size * num_nodes, seq_len, num_features)

        # LSTM 处理 [batch * nodes, seq_len, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 获取最后一个时间步的输出 [batch * nodes, hidden_size]
        # 也可以用 h_n.squeeze(0) 来获取隐状态
        output = lstm_out[:, -1, :]
        pm25 = self.fc_pm25(output)
        pm25_out = pm25.view(batch_size, num_nodes)

        return pm25_out


class BSiLU(nn.Module):
    def __init__(self, alpha: float = 1.667):
        """
        B-SiLU 激活函数模块
        :param alpha: 公式中的超参数，默认1.667
        """
        super().__init__()
        self.alpha = alpha  # 固定超参数（非可学习）
        self.alpha_half = self.alpha / 2  # 预计算α/2，避免重复计算

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：计算B-SiLU(x)"""
        sigmoid = torch.sigmoid(x)  # σ(x)
        return (x + self.alpha) * sigmoid - self.alpha_half


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, nodes_per_batch):
        super().__init__()
        self.nodes_per_batch = nodes_per_batch  # 每个批次的节点数
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True  # 输入/输出为 [batch, seq, features]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x 形状: [batch , nodes, hidden_dim]

        # 应用自注意力 (Q=K=V)
        attn_output, _ = self.attention(x, x, x)

        res = x + attn_output  # 残差连接
        norm_res = self.norm(res)  # 层归一化

        return norm_res


class MultiTaskLoss(nn.Module):
    def __init__(self, num=3):
        super(MultiTaskLoss, self).__init__()
        params = torch.tensor([2.0, 1.0, 1.0], requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + self.params[i]
        return loss_sum

