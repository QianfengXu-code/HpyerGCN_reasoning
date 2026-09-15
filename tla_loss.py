import torch
import os

os.environ["OMP_NUM_THREADS"] = "3"

import torch.nn as nn
import random
import config as cfg
from sklearn.cluster import KMeans


def info_nce_loss(query, positive, contrastive_set, temperature=0.5, negative_nums=-1):
    """
    计算 InfoNCE 损失（公式(11)）

    参数:
        query: 查询向量 (shape: [batch, dim])，如 t_b 或 t_k
        positive: 正例向量 (shape: [batch, dim])，如 t_k 或 t_b
        contrastive_set: 对比集 (shape: [batch, batch, dim])，包含同批次所有样本的对应步骤表示
                         （例如 Bk 或 Bb，需扩展为 [batch, batch, dim] 以匹配每个查询的对比集）
        temperature: 温度系数 τ

    返回:
        loss: InfoNCE 损失值 (标量)
    """
    batch_size, dim = query.shape

    # 相似度计算（点积）：query 与对比集中所有样本的相似度
    # query: [batch, dim] → [batch, 1, dim]
    # contrastive_set: [batch, batch, dim] → [batch, dim, batch]
    # 相似度矩阵: [batch, 1, dim] @ [batch, dim, batch] → [batch, 1, batch] → [batch, batch]

    # torch.bmm()输入的两张量必须是三维张量，第一维必须大小相同，第二维和第三位必须满足矩阵乘法

    # sim [batch, batch] 表示[样本数量(对应查询), 样本数量(对应每个样本的相似度)]
    # sim.sum(dim=1) → [batch, 1] 将所有样本的相似度相加
    sim = torch.bmm(query.unsqueeze(1), contrastive_set.transpose(1, 2)).squeeze(1)  # [batch, batch]
    sim = torch.clamp(sim, min=-1.0 + 1e-6, max=1.0 - 1e-6)  # 避免exp(过大值)
    exp_sim = torch.exp(sim / temperature)

    sum_exp = exp_sim.sum(dim=1, keepdim=True)  # [batch, 1]
    safe_sum = torch.clamp(sum_exp, min=1e-8)  # 避免除以0

    # 正例掩码：每个查询的正例在对比集中的索引（对角线，因为 contrastive_set[i] 是第 i 个样本的表示）
    # 例如，第 i 个样本的正例是 contrastive_set[i][i]（即 positive[i]）
    if negative_nums == -1:
        positive_mask = torch.eye(batch_size, device=query.device, dtype=torch.bool)
    else:
        positive_mask = torch.zeros_like(sim, dtype=torch.bool)
        positive_mask[:, 0] = True  # 第0列是正例（同簇轨迹）

    # 计算 InfoNCE 损失：对每个样本，正例的 log(softmax(sim)) 均值取负
    # sim[positive_mask] [batch] 取对角线元素(正样本)
    result = exp_sim[positive_mask] / safe_sum.squeeze(1)
    result = torch.clamp(result, min=1e-8)
    loss = -torch.log(result).mean()
    return loss


def step_level_alignment_loss(Bk, Bb, k, temperature=0.5):
    """
    计算步骤级对齐损失（公式(14)）
    Bb中的b参数的采样应该在之前

    参数:
        Bk: 所有样本的第 k 步推理表示 (shape: [batch, dim]) → 对应对比集 B_k
        Bb: 所有样本的第 b 步推理表示 (shape: [batch, dim]) → 对应对比集 B_b
        k: 总推理步数（用于采样中间步骤 b）
        temperature: InfoNCE 温度系数 τ

    返回:
        L_SLA: 步骤级对齐损失 (标量)
    """
    batch_size, dim = Bk.shape

    # ---------------------- 步骤1：采样中间步骤 b ----------------------
    if k <= 1:
        raise ValueError("总步数 k 必须 > 1 才能采样中间步骤 b")
    # （注：实际训练中，b 应在每个 batch 前采样，此处简化为函数内采样）

    # ---------------------- 步骤2：准备查询和对比集 ----------------------
    # query1: t_b (第 b 步中间状态)，即 Bb 中的样本；positive1: t_k (第 k 步最终状态)，即 Bk 中的样本
    query1 = Bb  # [batch, dim]
    positive1 = Bk  # [batch, dim]
    # 对比集 B_k: 同批次所有样本的第 k 步表示 → [batch, batch, dim]（每个样本的对比集都是整个 Bk）
    contrastive_set_k = Bk.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, batch, dim]

    # query2: t_k (第 k 步最终状态)；positive2: t_b (第 b 步中间状态)
    query2 = Bk  # [batch, dim]
    positive2 = Bb  # [batch, dim]
    # 对比集 B_b: 同批次所有样本的第 b 步表示 → [batch, batch, dim]
    contrastive_set_b = Bb.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, batch, dim]

    # ---------------------- 步骤3：计算双向 InfoNCE 损失 ----------------------
    # F(t_b, t_k, B_k)：以 t_b 为查询，t_k 为正例，B_k 为对比集
    loss1 = info_nce_loss(query1, positive1, contrastive_set_k, temperature)
    # F(t_k, t_b, B_b)：以 t_k 为查询，t_b 为正例，B_b 为对比集
    loss2 = info_nce_loss(query2, positive2, contrastive_set_b, temperature)

    # 步骤级对齐损失：双向损失平均（公式(14)）
    L_SLA = 0.5 * (loss1 + loss2)
    return L_SLA


def trajectory_level_alignment_loss(
        h: torch.Tensor,
        y: torch.Tensor,
        num_clusters: int = 8,
        num_negatives: int = 8,
        temperature: float = 0.2
) -> torch.Tensor:
    """
    轨迹级对齐损失（基于真值聚类的正负样本划分）

    参数：
        h: 轨迹表示矩阵，形状为 [B*N, D]
            - B: Batch Size（批量大小）
            - N: Nodes（轨迹中的节点数量）
            - D: Dimension（表示维度）
        y: 真值矩阵，形状为 [B*N, 1]（如用户评分、行为强度）
        num_clusters: KMeans聚类的簇中心数量（正样本簇的数量）
        num_negatives: 每个查询的负例数量（从不同簇中采样）
        temperature: InfoNCE损失的温度系数（控制相似度的区分度）

    返回：
        loss: 轨迹级对齐损失（标量，值越小表示对齐效果越好）
    """
    # ---------------------- 步骤1：用KMeans对真值进行聚类，得到轨迹标签 ----------------------
    # 将真值转换为numpy数组（sklearn的KMeans需要）
    y_np = y.reshape(-1, 1).cpu().numpy()  # 形状从 [B*N, 1] 转换为 [B*N]
    # 初始化KMeans模型（簇中心数量为num_clusters）
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # 拟合真值数据，得到每个轨迹的簇标签（形状：[B*N]）
    labels = kmeans.fit_predict(y_np)
    # 将标签转换为PyTorch张量（与表示矩阵h同设备）
    labels = torch.tensor(labels, device=h.device)

    # ---------------------- 步骤2：为每个轨迹构造正负样本对 ----------------------
    batch_size = h.shape[0]  # 轨迹总数（B*N）
    pos_indices = []  # 存储每个轨迹的正例索引（同簇）
    neg_indices = []  # 存储每个轨迹的负例索引（不同簇）

    for i in range(batch_size):
        # 1. 正例：从同簇轨迹中随机选1个（排除自己）
        pos_mask = (labels == labels[i]) & (torch.arange(batch_size).to(cfg.device) != i)  # 同簇且非自身的掩码
        if pos_mask.sum() == 0:
            # 极端情况：无同簇轨迹，选自己作为正例（不推荐，但避免错误）
            pos_idx = i
        else:
            # 随机采样1个同簇轨迹的索引
            pos_idx = torch.multinomial(pos_mask.float(), 1).item()
        pos_indices.append(pos_idx)

        # 2. 负例：从不同簇轨迹中随机选num_negatives个
        neg_mask = (labels != labels[i])  # 不同簇的掩码
        if neg_mask.sum() < num_negatives:
            # 极端情况：负例不足，重复采样（不推荐， but avoid error）
            neg_idx = torch.multinomial(neg_mask.float(), num_negatives, replacement=True).tolist()
        else:
            # 随机采样num_negatives个不同簇轨迹的索引
            neg_idx = torch.multinomial(neg_mask.float(), num_negatives).tolist()
        neg_indices.append(neg_idx)

    # ---------------------- 步骤3：构造对比集（正例+负例） ----------------------
    # 正例表示：每个轨迹的正例表示（形状：[B*N, D]）
    h_pos = h[torch.tensor(pos_indices, device=h.device)]
    # 负例表示：每个轨迹的负例表示（形状：[B*N, num_negatives, D]）
    h_neg = h[torch.tensor(neg_indices, device=h.device)]
    # 对比集：正例（1个）+ 负例（num_negatives个）（形状：[B*N, 1+num_negatives, D]）
    contrastive_set = torch.cat([h_pos.unsqueeze(1), h_neg], dim=1)

    # ---------------------- 步骤4：计算InfoNCE损失 ----------------------
    # 查询向量：轨迹表示（形状：[B*N, D] → [B*N, 1, D]）
    query = h
    # 对比集转置：适应矩阵乘法（形状：[B*N, 1+num_negatives, D] → [B*N, D, 1+num_negatives]）
    positive = contrastive_set[:, 0, :]
    negative = contrastive_set[:, 1:, :]

    loss = info_nce_loss(query, positive, negative, temperature, num_negatives)

    # contrastive_set_t = contrastive_set.transpose(1, 2)
    # # 相似度计算（点积）：query与对比集的相似度（形状：[B*N, 1, 1+num_negatives] → [B*N, 1+num_negatives]）
    # sim = torch.bmm(query, contrastive_set_t).squeeze(1) / temperature
    #
    # # 正例掩码：标记对比集中的正例位置（第0列是正例）
    # positive_mask = torch.zeros_like(sim, dtype=torch.bool)
    # positive_mask[:, 0] = True  # 第0列是正例（同簇轨迹）
    #
    # # InfoNCE损失：-log(softmax(sim)[正例])的均值
    # # sim[positive_mask]：每个轨迹的正例相似度（形状：[B*N]）
    # # sim.sum(dim=1)：每个轨迹的总相似度（正例+负例，形状：[B*N]）
    # loss = -torch.log(sim[positive_mask] / sim.sum(dim=1)).mean()

    return loss
