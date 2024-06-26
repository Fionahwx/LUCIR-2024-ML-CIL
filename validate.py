import enum
import torch
import torch.nn as nn
from utils.AverageMeter import AverageMeter
from utils.utils import accuracy

# 验证函数
def validate(net, val_loader):
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化准确率和损失的计量器
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    # 将网络移到GPU并设置为评估模式
    net.cuda()
    net.eval()
    # 在不计算梯度的情况下进行验证
    with torch.no_grad():
        for batch_id, (inputs, targets, t) in enumerate(val_loader):
            # 将输入和目标移到GPU
            inputs, targets = inputs.cuda(), targets.cuda()
            # 获取网络的输出
            _, output = net(inputs)
            # 计算验证集的准确率
            acc_val = accuracy(output, targets, topk=(1,))
            
            # 计算损失
            loss = criterion(output, targets.long())
            # 更新计量器
            acc_meter.update(acc_val[0].item(), inputs.size(0))
            loss_meter.update(loss.item(), inputs.size(0))
    # 返回平均准确率和平均损失
    return acc_meter.avg, loss_meter.avg

# 提取特征函数
def extract_features(args, net, loader):
    features = None
    # 将网络移到GPU并设置为评估模式
    net.cuda()
    net.eval()

    # 在不计算梯度的情况下提取特征
    with torch.no_grad():
        for inputs, targets in loader:
            # 将输入移到GPU
            inputs = inputs.cuda()
            # 获取网络的特征
            f, _ = net(inputs)
            # 将特征连接起来
            if features is not None:
                features = torch.cat((features, f), 0)
            else:
                features = f

    # 返回特征并将其移到CPU
    return features.detach().cpu().numpy()

# L2标准化函数
def l2_norm(input, axis=1):
    # 计算L2范数
    norm = torch.norm(input, 2, axis, True)
    # 进行标准化
    output = torch.div(input, norm)
    return output
