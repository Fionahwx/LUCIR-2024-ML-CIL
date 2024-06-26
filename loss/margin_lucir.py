import torch
from torch._C import dtype
import torch.nn as nn

def ucir_ranking(logits, targets, task_size, nb_negatives=2, margin=0.2):
    """Hinge loss from UCIR.

    Taken from: https://github.com/hshustc/CVPR19_Incremental_Learning

    # References:
        * Learning a Unified Classifier Incrementally via Rebalancing
          Hou et al.
          CVPR 2019
    """

    # gt_index = torch.zeros(logits.size()).to(logits.device)
    # gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5) # One-hot encoding

    # 将目标转换为long类型并移到与logits相同的设备上
    targets = targets.long().to(logits.device)
    
    # 创建一个与logits相同大小的零张量
    gt_index = torch.zeros(logits.size(), device=logits.device)
    
    # 使用scatter方法将targets转换为one-hot编码
    gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
    
    # 获取正确类的分数
    gt_scores = logits.masked_select(gt_index)  # 获取新类上的最高分数

    # 计算旧类的数量
    num_old_classes = logits.shape[1] - task_size
    
    # 获取新类中的top-K负样本的分数
    max_novel_scores = logits[:, num_old_classes:].topk(nb_negatives, dim=1)[0]  # 硬样本（旧类样本）的索引
    
    # 找到属于旧类的目标索引
    hard_index = targets.lt(num_old_classes)
    
    # 计算硬样本的数量
    hard_num = torch.nonzero(hard_index).size(0)

    if hard_num > 0:
        # 将正确类的分数重复nb_negatives次
        gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, nb_negatives)
        max_novel_scores = max_novel_scores[hard_index]
        
        # 确保正确类的分数和负样本的分数形状匹配
        assert (gt_scores.size() == max_novel_scores.size())
        assert (gt_scores.size(0) == hard_num)
        
        # 计算Margin Ranking Loss
        loss = nn.MarginRankingLoss(margin=margin)(
            gt_scores.view(-1, 1), 
            max_novel_scores.view(-1, 1), 
            torch.ones(hard_num * nb_negatives).to(logits.device)
        )
        return loss

    # 如果没有硬样本，返回0损失
    return torch.tensor(0).float()