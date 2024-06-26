import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

# 定义余弦相似度分类器模型
class CosineClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        # 定义模型参数
        self.weight = Parameter(torch.Tensor(n_classes, input_dim))  # 权重参数，形状为 (类别数, 输入维度)
        self.eta = Parameter(torch.Tensor(1))  # 缩放参数，用于调整余弦相似度的范围
        # 初始化模型权重
        self.init_weights()
        
    # 初始化权重
    def init_weights(self):
        stdv = 1. / math.sqrt(self.weight.size(1))  # 根据输入维度计算标准差
        self.weight.data.uniform_(-stdv, stdv)  # 用均匀分布初始化权重
        self.eta.data.fill_(1)  # 将缩放参数初始化为1，可以根据需要进行调整

    # 前向传播函数
    def forward(self, x):
        # 对输入特征和权重进行 L2 归一化，将它们投射到单位超球面上
        x_norm = F.normalize(x, p=2, dim=1)  # 对输入进行归一化
        w_norm = F.normalize(self.weight, p=2, dim=1)  # 对权重进行归一化
        # 计算缩放后的余弦相似度分数，并进行线性变换
        y = self.eta * F.linear(x_norm, w_norm)  # 使用缩放参数乘以线性变换结果
        return y  # 返回分类得分

    