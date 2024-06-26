from os import W_OK
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
from utils.AverageMeter import AverageMeter
from utils.utils import extract_batch_from_memory, accuracy, save_model
from loss.less_forget import EmbeddingsSimilarity
from loss.margin_lucir import ucir_ranking

def train(args, loader_train, net, task_id, criterion_cls, previous_net, optimizer, epoch, lamda, old_classes):
    # 初始化准确率和损失的计量器
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    # 设置网络为训练模式
    net.train()
    # 遍历训练数据加载器中的每个批次
    for batch_id, (inputs, targets, t) in enumerate(loader_train):
        # 将输入和目标移到GPU
        inputs, targets = inputs.cuda(), targets.cuda()
        # 获取特征和输出
        feature, output = net(inputs)
        # 计算分类损失
        loss = criterion_cls(output, targets.long())
        # 如果当前任务不是第一个任务且存在之前的网络
        if task_id > 0 and previous_net is not None:
            with torch.no_grad():
                # 获取旧网络的特征和输出
                feature_old, output_old = previous_net(inputs)
            # 如果启用了less forgetting loss
            if args.less_forg:
                # 计算less forgetting loss
                loss_less_forget = EmbeddingsSimilarity(l2_norm(feature_old), l2_norm(feature))
                loss += lamda * loss_less_forget
            # 如果启用了ranking loss
            if args.ranking:
                # 创建mask以过滤旧类
                mask = [False if i in old_classes else True for i in targets]
                # 计算ranking loss
                loss_margin = ucir_ranking(logits=output[mask],
                                           targets=targets[mask],
                                           task_size=args.increment,
                                           nb_negatives=args.knegatives,
                                           margin=args.ranking)
                # 将ranking loss加到总损失中
                loss += loss_margin
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新网络参数
        optimizer.step()
        # 计算训练准确率
        acc_training = accuracy(output, targets, topk=(1,))
        # 更新计量器
        acc_meter.update(acc_training[0].item(), inputs.size(0))
        loss_meter.update(loss.item(), inputs.size(0))
    
    # 打印训练结果
    print(f"TRAIN \t Epoch: {epoch}/{args.epochs}\t loss: {loss_meter.avg}\t acc: {acc_meter.avg}")

# 平衡微调函数
def balanced_finetuning(args, loader_train, net, task_id, criterion_cls, optimizer, epoch):
    # 初始化准确率和损失的计量器
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    # 设置网络为训练模式
    net.train()
    # 遍历训练数据加载器中的每个批次
    for batch_id, (inputs, targets) in enumerate(loader_train):
        # 将输入和目标移到GPU
        inputs, targets = inputs.cuda(), targets.cuda()
        # 获取输出
        _, output = net(inputs)
        # 计算分类损失
        loss = criterion_cls(output, targets)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新网络参数
        optimizer.step()
        # 计算训练准确率
        acc_training = accuracy(output, targets, topk=(1,))
        # 更新计量器
        acc_meter.update(acc_training[0].item(), inputs.size(0))
        loss_meter.update(loss.item(), inputs.size(0))
    # 打印平衡训练结果
    print(f"BALANCED TRAIN \t Epoch: {epoch}/{args.ft_epochs}\t loss: {loss_meter.avg}\t acc: {acc_meter.avg}")

# L2标准化函数
def l2_norm(input, axis=1):
    # 计算L2范数
    norm = torch.norm(input, 2, axis, True)
    # 进行标准化
    output = torch.div(input, norm)
    return output
