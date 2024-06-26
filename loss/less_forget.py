import torch
import torch.nn.functional as F  # 神经网络操作函数库

def EmbeddingsSimilarity(feature_a, feature_b):
    return F.cosine_embedding_loss(  # 计算余弦嵌入损失
        feature_a, feature_b,
        torch.ones(feature_a.shape[0]).to(feature_a.device)
    )
