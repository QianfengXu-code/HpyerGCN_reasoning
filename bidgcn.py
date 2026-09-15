import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # 可选的权重归一化（非必须）
        self.weight_norm = nn.Sequential(
            nn.Tanh(),  # 约束权重到[-1,1]范围（保留负值）
            nn.LayerNorm(out_features)  # 稳定训练
        )

    def forward(self, x, adj):
        """双向图卷积"""
        # 原始方向传播

        A_in_degree = adj.sum(dim=0)
        D_in = torch.diag(A_in_degree).unsqueeze(0).repeat(x.size(0), 1, 1)

        A_out_degree = adj.sum(dim=1)
        D_out = torch.diag(A_out_degree).unsqueeze(0).repeat(x.size(0), 1, 1)

        x_transformed = self.linear(x)
        if len(D_in.shape) == 3:  # 批量处理
            out_forward = torch.bmm(D_in, x_transformed)
            out_backward = torch.bmm(D_out, x_transformed)
        else:  # 单样本处理
            out_forward = torch.mm(adj, x_transformed)
            out_backward = torch.mm(adj.t(), x_transformed)

        # 合并双向信息（这里用相加，也可改用拼接）
        return self.weight_norm(out_forward + out_backward)


class DynamicBidirectionalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gc1 = BidirectionalGraphConvolution(input_dim, hidden_dim)
        self.gc2 = BidirectionalGraphConvolution(hidden_dim, output_dim)

    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        return self.gc2(h, adj)


# 测试用例
if __name__ == "__main__":
    batch_size = 12
    num_nodes = 24
    input_dim = 32
    hidden_dim = 64
    output_dim = 16

    model = DynamicBidirectionalGCN(input_dim, hidden_dim, output_dim)

    # 生成不对称且有负值的动态矩阵
    adj = torch.randn(num_nodes, num_nodes)  # [-1, 1]范围
    adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
    x = torch.randn(batch_size, num_nodes, input_dim)

    output = model(x, adj)
    print("输出形状:", output.shape)  # 应输出 [24, 16]