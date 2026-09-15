import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from utils import lorentz_inner_product, lorentz_norm, get_pole
import matplotlib.pyplot as plt


class HyperbolicOperations:
    def __init__(self, c=1.0, eps=1e-5, euclid_dim=64, device='cpu'):
        """
        初始化双曲几何运算器
        :param c: 曲率参数 (默认为1.0)
        :param eps: 数值稳定性常数
        """
        self.c = torch.tensor(c, dtype=torch.float32, device=device)
        self.eps = torch.tensor(eps, dtype=torch.float32, device=device)
        self.o = torch.tensor(np.zeros(euclid_dim + 1), dtype=torch.float32, device=device)  # 扩展欧式空间的特征，并+1， [d + 1]
        self.o[0] = torch.sqrt(self.c)  # 定义北极点[sqrt(k), 0, 0, 0,...]

        self.pole = torch.tensor(np.zeros((euclid_dim + 1) * 2), dtype=torch.float32,
                                 device=device)  # 扩展欧式空间的特征，并+1， [d + 1]
        self.pole[0] = torch.sqrt(self.c)  # 定义北极点[sqrt(k), 0, 0, 0,...]

    def log_map_x(self, x, y):
        """对数映射（修正内积符号）"""
        xy_inner = lorentz_inner_product(x, y)  # 内积 <x,y>_L

        # 双曲距离：d_L(x,y) = sqrt(K) * arccosh( -<x,y>_L / K )（因<x,x>_L=-K）
        arg = torch.clamp(-xy_inner / self.c, min=1.0 + 1e-10)  # 注意负号
        d_L = torch.sqrt(self.c) * torch.acosh(arg)

        # 分子：y + (1/K)*<x,y>_L * x（内积符号已修正）
        numerator = y + (xy_inner / self.c).unsqueeze(-1) * x
        denominator = lorentz_norm(numerator)
        denominator_clamped = torch.clamp(denominator, min=1e-10)

        return (d_L / denominator_clamped).unsqueeze(-1) * numerator

    def exp_map_x(self, x, v):
        """指数映射（修正内积符号）"""
        sqrt_K = torch.sqrt(self.c)
        v_norm = lorentz_norm(v)  # 切空间向量范数（正定）
        v_norm_clamped = torch.clamp(v_norm, min=1e-10)

        t = v_norm_clamped / sqrt_K
        cosh_t = torch.cosh(t)
        sinh_t = torch.sinh(t)

        # 公式：cosh(t)*x + sqrt(K)*sinh(t)/v_norm * v（x满足<x,x>_L=-K）
        term1 = cosh_t.unsqueeze(-1) * x
        term2 = (sqrt_K * sinh_t / v_norm_clamped).unsqueeze(-1) * v
        return term1 + term2

    def poincare_distance(self, u, v):
        """
        计算庞加莱圆盘模型中两点间的双曲距离
        """
        # 确保输入在单位圆盘内
        u = self.project_to_ball(u)
        v = self.project_to_ball(v)

        # 计算欧几里得距离平方
        uv = torch.sum((u - v) ** 2, dim=-1)
        u_norm_sq = torch.sum(u ** 2, dim=-1).clamp(min=self.eps, max=1 - self.eps)
        v_norm_sq = torch.sum(v ** 2, dim=-1).clamp(min=self.eps, max=1 - self.eps)

        # 庞加莱距离公式
        alpha = (1 - u_norm_sq)
        beta = (1 - v_norm_sq)
        gamma = 1 + 2 * uv / (alpha * beta)

        # 使用数值稳定的反双曲余弦函数
        # 增加一个小的epsilon防止gamma=1时出现log(0)
        gamma = gamma.clamp(min=1 + self.eps)
        distance = torch.acosh(gamma) / torch.sqrt(torch.tensor(self.c))

        return distance

    def expmap(self, x, v):
        """
        在庞加莱圆盘模型中的指数映射
        :param x: 切点 (在双曲空间)
        :param v: 切向量 (在欧几里得空间)
        """
        x = self.project_to_ball(x)  # 将坐标投影到圆盘，并对非法点（不满足圆盘定义的点）进行修正
        # clamp，截断小于这个数的值，将其变为self.eps
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp(min=self.eps, max=1 - self.eps)

        # 计算lambda_x，双曲空间局部缩放因子
        lambda_x = 2 / (1 - self.c * x_norm_sq)

        # 方向向量，使用L2范数进行归一化处理
        direction = F.normalize(v, p=2, dim=-1)

        # 指数映射计算
        term1 = torch.cosh(torch.sqrt(self.c) * lambda_x * v_norm / 2)
        term2 = torch.sinh(torch.sqrt(self.c) * lambda_x * v_norm / 2)

        # 结果计算
        result = self.mobius_add(
            x,
            direction * term2 / (torch.sqrt(self.c) * term1)
        )

        return self.project_to_ball(result)

    def logmap(self, x, y):
        """
        在庞加莱圆盘模型中的对数映射
        :param x: 切点 (在双曲空间)
        :param y: 目标点 (在双曲空间)
        """
        x = self.project_to_ball(x)
        y = self.project_to_ball(y)
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=self.eps)

        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp(min=self.eps, max=1 - self.eps)
        lambda_x = 2 / (1 - self.c * x_norm_sq)

        # 对数映射计算
        scale = 1 / (torch.sqrt(self.c) * lambda_x)
        result = scale * torch.acosh(1 + 2 * diff_norm ** 2 / (
                (1 - self.c * x_norm_sq) * (1 - self.c * torch.sum(y ** 2, dim=-1, keepdim=True))
        )) * F.normalize(diff, p=2, dim=-1)

        return result

    def mobius_add(self, x, y):
        """
        庞加莱圆盘模型中的Möbius加法
        """
        x = self.project_to_ball(x)
        y = self.project_to_ball(y)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp(min=self.eps, max=1 - self.eps)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True).clamp(min=self.eps, max=1 - self.eps)

        # Möbius加法公式
        numerator = (1 + 2 * self.c * xy + self.c * y_norm_sq) * x + (1 - self.c * x_norm_sq) * y
        denominator = 1 + 2 * self.c * xy + self.c ** 2 * x_norm_sq * y_norm_sq

        return self.project_to_ball(numerator / denominator.clamp(min=self.eps))

    def mobius_matvec(self, m, x):
        """
        双曲空间中的矩阵乘法 (Möbius变换)
        :param m: 权重矩阵 (欧几里得空间)
        :param x: 输入点 (双曲空间)
        """
        # 将对数映射应用到原点
        log_x = self.logmap(torch.zeros_like(x), x)

        # 在切空间进行线性变换
        transformed = torch.matmul(log_x, m.t())

        # 指数映射回双曲空间
        return self.expmap(torch.zeros_like(x), transformed)

    def project_to_ball(self, x):
        """
        将点投影到单位圆盘内，确保数值稳定性
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)  # 对最外的维度求2范数
        mask = (norm >= 1.0).float()
        # 对于范数>=1的点进行投影，保持数值稳定性
        projected = x / (norm + self.eps)  # eps 为一个很小的数
        return (1 - mask) * x + mask * (projected * (1 - self.eps))

    def hyperbolic_activation(self, x, activation=torch.relu):
        """
        双曲激活函数
        """
        # 映射到切空间
        tangent = self.logmap(torch.zeros_like(x), x)
        # 应用激活函数
        activated_tangent = activation(tangent)
        # 映射回双曲空间
        return self.expmap(torch.zeros_like(x), activated_tangent)


# ====================== 双曲图卷积层 ======================
class HyperbolicGraphConv(nn.Module):
    def __init__(self, in_features, out_features, hyp_ops, ctype='normal', bias=True, device='cpu'):
        super(HyperbolicGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.hyp_ops = hyp_ops
        self.ctype = ctype
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).to(device))

        if bias:
            # 需要留一位0维赋值为0， 保证切向量与原点相切
            self.bias = nn.Parameter(torch.Tensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.mlp_attn = nn.Sequential(
            nn.Linear(2 * out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, 1)
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """投影到双曲面流形（确保 <x,x>_L = -K）"""
        x_rest = x[..., 1:]
        x0_sq = self.hyp_ops.c + torch.sum(x_rest ** 2, dim=-1, keepdim=True)
        x0 = torch.sqrt(torch.clamp(x0_sq, min=1e-10))
        return torch.cat([x0, x_rest], dim=-1)

    def hyper_agg(self, x, i_indices, j_indices, pole):

        nodes, d = x.shape

        # adj_mask = adj.bool()
        x_tangent = self.hyp_ops.log_map_x(pole, x)  # 将所有节点转移到切向量上

        # upper_tri_mask = torch.triu(adj_mask, diagonal=1)  # 上三角掩码（不含对角线）
        # i_indices, j_indices = torch.where(upper_tri_mask)  # 数量相同， 相对应的位置存储邻接边的索引

        x_i = x_tangent[i_indices]  # [E//2, d]
        x_j = x_tangent[j_indices]  # [E//2, d]
        x_ij = torch.cat([x_i, x_j], dim=1)  # [E//2, 2d] 链接
        attn_scores = self.mlp_attn(x_ij).squeeze(1)  # [E//2] mlp计算未归一化分数

        i_bidir = torch.cat([i_indices, j_indices], dim=0)  # [E]
        j_bidir = torch.cat([j_indices, i_indices], dim=0)  # [E]
        attn_weights_bidir = torch.cat([attn_scores, attn_scores], dim=0)  # [E]（双向共享权重）

        x_h_agg = []
        for i in range(nodes):

            x_i_h = x[i].unsqueeze(0)  # [1, d+1=33] 此处应该是用当前节点的特征,不是映射到切空间的特征

            # 找到节点i的所有邻接节点j（含双向边）
            j_mask = (i_bidir == i)
            if not torch.any(j_mask):
                # 无邻接节点时，聚合结果为自身???? 再纠正一下
                x_h_agg.append(x[i].squeeze(0))
                continue

            # 收集邻接节点j的双曲特征和对应权重
            j_nodes = j_bidir[j_mask]  # [M]（M为i的邻接节点数）获取当前节点的邻接点
            w = attn_weights_bidir[j_mask]  # [M]
            w_softmax = F.softmax(w, dim=0)

            # 计算 log_{x_i^H}(x_j^H)（x_j在x_i切空间的向量）
            x_j_h = x[j_nodes]  # [M, d+1=33] 当前节点的邻接节点的特征
            log_xj_xi = self.hyp_ops.log_map_x(x_i_h, x_j_h)  # [M, 33] 然后再用对数映射返回到切空间

            # 加权求和（切空间向量）
            log_agg = torch.sum(w_softmax.unsqueeze(-1) * log_xj_xi, dim=0, keepdim=True)  # [1, 33]

            # 指数映射回双曲面
            x_i_agg = self.hyp_ops.exp_map_x(x_i_h, log_agg)  # [1, 33]
            x_i_agg_re = self.project_to_manifold(x_i_agg)  # 确保在双曲面上
            x_h_agg.append(x_i_agg_re.squeeze(0))

        x_h_agg = torch.stack(x_h_agg, dim=0)  # [64, 33]
        return x_h_agg

    def hyper_activation(self, x_hyper):
        nodes, d = x_hyper.shape

        pole = get_pole(d, self.hyp_ops.c)

        x_tangent = self.hyp_ops.log_map_x(pole, x_hyper)
        x_sigmoid = torch.sigmoid(x_tangent)
        x_hyper_activated = self.hyp_ops.exp_map_x(pole, x_sigmoid)

        return x_hyper_activated

    def forward(self, x_hyper_0, i_indices, j_indices, hyp_tools):
        """
        x: 双曲空间中的节点特征 [num_nodes, in_features]
        adj: 邻接矩阵 [num_nodes, num_nodes]
        """
        nodes, d = x_hyper_0.shape
        feature_0, feature_1 = self.weight.shape  # 获取特征变换前后的特征数

        pole_0 = get_pole(feature_0, self.hyp_ops.c)
        pole_1 = get_pole(feature_1, self.hyp_ops.c)

        # 线性变换
        x_tangent_0 = hyp_tools.log_map_x(pole_0, x_hyper_0)
        x_tangent_1 = torch.matmul(x_tangent_0, self.weight)
        x_hyper_1_no_bias = hyp_tools.exp_map_x(pole_1, x_tangent_1)
        x_non_agg_wo_manifold = torch.add(x_hyper_1_no_bias, self.bias)
        x_non_agg = self.project_to_manifold(x_non_agg_wo_manifold)  # 强制转换到双曲面上

        x_agg = self.hyper_agg(x_non_agg, i_indices, j_indices, pole_1)  # 双曲聚合操作
        x_agg_activated = self.hyper_activation(x_agg)

        # # 组合张量， 获得符合定义的切向量表示
        # bias = torch.cat([torch.tensor([0], dtype=torch.float32, device=self.device), self.bias])
        #
        # tangent_xh = hyp_tools.log_map_x(self.hyp_ops.pole, x_hyper_1_no_bias)
        # tangent_oo = hyp_tools.log_map_x(x_hyper_1_no_bias, self.hyp_ops.pole)
        # xb_inner_product = lorentz_inner_product(tangent_xh, bias)
        # ox_inner_product = lorentz_inner_product(self.hyp_ops.pole, x_hyper_1_no_bias)
        # arg = torch.clamp(-ox_inner_product / self.hyp_ops.c, min=1.0 + 1e-10)
        # d_L = torch.clamp(torch.sqrt(self.hyp_ops.c) * torch.acosh(arg), min=1e-10)
        #
        # transfer_1 = xb_inner_product / torch.pow(d_L, 2)
        # transfer_2 = torch.add(tangent_xh, tangent_oo)
        #
        # transfer_result = bias - torch.matmul(transfer_1, transfer_2)

        return x_agg_activated


class HGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim, curvature=1.0, ctype='sid', depth=3, device='cpu'):
        super(HGCN, self).__init__()
        self.hyp_ops = HyperbolicOperations(c=curvature, euclid_dim=in_features, device=device)
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.hyper_dim = in_features + 1
        self.dim_fc = nn.Linear(in_features, hidden_dim - 1)
        self.conv = []

        if ctype == 'sid':
            self.conv.append(
                HyperbolicGraphConv(hidden_dim, hidden_dim, self.hyp_ops, ctype=ctype,
                                    device=device).to(device))
            self.conv.append(
                HyperbolicGraphConv(hidden_dim, hidden_dim, self.hyp_ops, ctype=ctype,
                                    device=device).to(device))
            self.conv.append(
                HyperbolicGraphConv(hidden_dim, hidden_dim, self.hyp_ops, ctype=ctype,
                                    device=device).to(device))

        elif ctype == 'bid':
            self.conv = HyperbolicGraphConv(hidden_dim, hidden_dim, self.hyp_ops, ctype=ctype)


    @staticmethod
    def initial_from_euclid(self, x_euclid, curvature):

        sqrt_k = torch.sqrt(curvature)
        x_norm = torch.norm(x_euclid, p=2, dim=1, keepdim=True)  # 计算2范数，形状: (N, 1)
        x_norm = torch.clamp_min(x_norm, min=1e-10)  # 防止零范数导致除零错误

        # 第一个分量
        cosh_arg = x_norm / sqrt_k  # 形状: (N, 1)
        x0 = sqrt_k * torch.cosh(cosh_arg)  # 形状: (N, 1)

        # 公式(5)的后d个分量：sqrt(K) * sinh(||x_e|| / sqrt(K)) * (x_e / ||x_e||)
        sinh_arg = x_norm / sqrt_k  # 形状: (N, 1)
        sinh_term = sqrt_k * torch.sinh(sinh_arg)  # 形状: (N, 1)
        direction = x_euclid / x_norm  # 单位化方向向量，形状: (N, d)
        x_rest = sinh_term * direction  # 形状: (N, d)

        # 拼接第一个分量和后d个分量，得到双曲特征 (N, d+1)
        # 直接用公式，就不用指数映射了
        x_hyper = torch.cat([x0, x_rest], dim=1)

        return x_hyper

    def forward(self, x, i_indices, j_indices):
        # 残差连接版本
        # 是否需要线性变换？??
        # 双曲空间下初始节点表示
        x_fc = self.dim_fc(x)
        hyper_feat = self.initial_from_euclid(self, x_euclid=x_fc, curvature=self.hyp_ops.c)  # [nodes, d + 1]
        # # 通过对数映射映射到切空间
        # x_tagent = self.hyp_ops.log_map_x(self.hyp_ops.o, hyper_feat)  # 对数映射测试
        # # 通过指数映射到双曲空间
        # x = self.hyp_ops.exp_map_x(self.hyp_ops.o, x_tagent)

        final_embs = [hyper_feat]
        for layer_idx in range(self.depth):
            embs = self.conv[layer_idx](hyper_feat, i_indices, j_indices, self.hyp_ops)  # x是输入特征，adj是邻接图
            embs = embs + final_embs[-1]
            final_embs.append(embs)
        # final_embs_tensor = torch.stack(final_embs)
        # att_embs = self.att(final_embs_tensor)
        final_embs = torch.mean(torch.stack(final_embs), dim=0)

        # pole = get_pole(self.hidden_dim, self.hyp_ops.c)
        #
        # final_embs_tangent = self.hyp_ops.log_map_x(final_embs, pole)

        # 是否需要返回切向量的结果？？？

        return final_embs

    # def forward(self, x, adj):
    #     # 欧式空间到双曲空间的初始映射 线性层-> 1000,16
    #     eucl_feat = self.initial_map(x)
    #     # 通过指数映射到双曲空间
    #     x = self.hyp_ops.expmap(torch.zeros_like(eucl_feat), eucl_feat)
    #
    #     # 双曲图卷积层
    #     x = self.hyp_ops.hyperbolic_activation(self.conv1(x, adj))
    #     x = self.conv2(x, adj)
    #
    #     return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
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
        residual = x.mean(dim=0)
        att = attn_output.mean(dim=0)
        res = residual + att  # 残差连接
        norm_res = self.norm(res)  # 层归一化

        return norm_res


def prepare_poi_data(coordinates, k_neighbors=10):
    """
    准备POI数据：构建k近邻图
    :param coordinates: POI经纬度坐标 [num_poi, 2]
    :param k_neighbors: 每个点的邻居数量
    :return: 特征矩阵和邻接矩阵
    """
    # 转换为PyTorch张量
    features = torch.tensor(coordinates, dtype=torch.float32)

    # 构建k近邻图
    adj = kneighbors_graph(coordinates, k_neighbors, mode='connectivity', include_self=False)  # 输出为稀疏矩阵
    adj = adj + adj.T  # 使邻接矩阵对称
    adj[adj > 1] = 1  # 确保没有多重边

    # 转换为PyTorch稀疏张量
    adj = adj.tocoo()
    indices = torch.tensor([adj.row, adj.col], dtype=torch.long)
    values = torch.tensor(adj.data, dtype=torch.float32)
    adj_sparse = torch.sparse_coo_tensor(indices, values, adj.shape)

    return features, adj_sparse


# ====================== 自定义损失函数 ======================
class HyperbolicDistanceLoss(nn.Module):
    def __init__(self, hyp_ops):
        super(HyperbolicDistanceLoss, self).__init__()
        self.hyp_ops = hyp_ops

    def forward(self, embeddings, adj):
        """
        基于原始图中邻居关系在双曲空间中的距离损失
        """
        loss = 0.0
        indices = adj._indices()
        values = adj._values()

        for i in range(indices.shape[1]):
            src = indices[0, i]
            dst = indices[1, i]
            weight = values[i]

            # 计算双曲距离
            dist = self.hyp_ops.poincare_distance(embeddings[src], embeddings[dst])

            # 损失计算：距离应接近边的权重
            loss += (dist - weight) ** 2

        return loss / indices.shape[1]
