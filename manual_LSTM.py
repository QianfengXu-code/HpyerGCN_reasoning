import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 输入门
        self.W_ii = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))

        # 遗忘门
        self.W_if = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))

        # 候选记忆
        self.W_ig = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(hidden_dim))

        # 输出门
        self.W_io = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        # 输入门
        i_t = torch.sigmoid(F.linear(x, self.W_ii, self.b_i) +
                            F.linear(h_prev, self.W_hi))

        # 遗忘门
        f_t = torch.sigmoid(F.linear(x, self.W_if, self.b_f) +
                            F.linear(h_prev, self.W_hf))

        # 候选记忆
        g_t = torch.tanh(F.linear(x, self.W_ig, self.b_g) +
                         F.linear(h_prev, self.W_hg))

        # 更新记忆单元
        c_t = f_t * c_prev + i_t * g_t

        # 输出门
        o_t = torch.sigmoid(F.linear(x, self.W_io, self.b_o) +
                            F.linear(h_prev, self.W_ho))

        # 更新隐藏状态
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class Reasoning_Cell(nn.Module):
    def __init__(self, dims):
        super(Reasoning_Cell, self).__init__()

        self.norm = nn.LayerNorm(2 * dims)
        self.core_mlp = nn.Sequential(
            nn.Linear(2 * dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims)
        )

    def forward(self, x_hidden, t_0):

        # f
        xt_concat = torch.cat([x_hidden, t_0], dim=-1)

        # LN 层归一化
        xt_norm = self.norm(xt_concat)

        # C 核心变化层
        t_1 = self.core_mlp(xt_norm)

        return t_1


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq=12, reason_dim=12, num_layers=1, reasoning_layers=5):
        super(CustomLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.reason_embs = nn.Embedding(seq, reason_dim)

        # 创建多层LSTM单元
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_dim if i == 0 else hidden_dim
            self.layers.append(CustomLSTMCell(input_size, hidden_dim))

        self.reasoning_layer = nn.ModuleList()
        for i in range(reasoning_layers):
            self.reasoning_layer.append(Reasoning_Cell(hidden_dim))

    @staticmethod
    def sample_t0(batch_nodes, d, sigma1=0.5):
        """
        采样初始思维状态 T0 ~ N(0, sigma1² I) (简洁版)
        参数:
            d: 向量维度
            sigma1: 标准差σ₁
        """
        # torch.randn(d) → N(0, 1)，乘以sigma1后 → N(0, sigma1²)
        t0_single = sigma1 * torch.randn(d)
        t0 = torch.stack([t0_single] * batch_nodes, dim=0)

        return t0

    def forward(self, x, pm25_feature, hidden=None):
        # x的形状: (batch*nodes, seq_len, input_dim)
        batch_nodes, seq_len, _ = x.size()
        reason_ith = self.reason_embs.weight.unsqueeze(0).repeat(batch_nodes, 1, 1)

        # 初始化隐藏状态
        if hidden is None:
            h = torch.zeros(self.num_layers, batch_nodes, self.hidden_dim, device=x.device)
            c = torch.zeros(self.num_layers, batch_nodes, self.hidden_dim, device=x.device)
        else:
            h, c = hidden

        x_with_reason = torch.cat([x, reason_ith], dim=-1)

        # 存储所有时间步的输出
        outputs = []

        # 按时间步处理
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch*nodes, input_dim)

            # 逐层处理
            new_h = []
            new_c = []
            for layer_idx, layer in enumerate(self.layers):
                h_t, c_t = layer(x_t, (h[layer_idx], c[layer_idx]))
                new_h.append(h_t)
                new_c.append(c_t)
                x_t = h_t  # 下一层的输入是当前层的输出

            # 更新当前层的隐藏状态、细胞状态
            h = torch.stack(new_h, dim=0)
            c = torch.stack(new_c, dim=0)

            h_reasoning = h.squeeze(0)

            # 仅隐式推理最后一步
            # reason_out存储所有隐式推理的状态t

            # 存储最后一层的输出
            outputs.append(x_t)

        # 将输出堆叠为(batch*nodes, seq_len, hidden_dim)
        outputs = torch.stack(outputs, dim=1)

        return outputs, (h, c)


if __name__ == "__main__":
    # 参数设置
    batch_nodes = 32  # batch_size * nodes
    seq_len = 10  # 时间序列长度
    input_dim = 64  # 输入特征维度
    hidden_dim = 128  # 隐藏层维度
    num_layers = 1  # LSTM层数

    # 创建模型
    lstm = CustomLSTM(input_dim, hidden_dim, num_layers)

    # 创建输入数据 (batch*nodes, seq_len, input_dim)
    x = torch.randn(batch_nodes, seq_len, input_dim)

    # 前向传播
    outputs, (h_n, c_n) = lstm(x)

    print("输入形状:", x.shape)
    print("输出形状:", outputs.shape)  # 应该是 (batch*nodes, seq_len, hidden_dim)
    print("最终隐藏状态形状:", h_n.shape)  # 应该是 (num_layers, batch*nodes, hidden_dim)
    print("最终记忆单元形状:", c_n.shape)  # 应该是 (num_layers, batch*nodes, hidden_dim)
