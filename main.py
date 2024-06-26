import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from models import create
from continuum.datasets import CIFAR100
from continuum import ClassIncremental
import models
from train import train, balanced_finetuning
from validate import validate
from utils.ExemplarSet import ExemplarSet
from utils.utils import get_all_images_per_class
from torch.optim.lr_scheduler import MultiStepLR
from utils.feature_selection import perform_selection
from utils.utils import getTransform, get_dataset, get_sampler, save_model, load_model
from torchvision.transforms import transforms

import os
import numpy as np
import copy

parser = argparse.ArgumentParser(description='Learning unified classifier via rebalancing')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default="cifar100", metavar='BATCH', help='dataset')
parser.add_argument('--start', type=int, default=50, help='初始类的数量 starting classes')
parser.add_argument('--increment', type=int, default=5, help='每个任务增加的类的数量increment classes at each task')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for cnn')
parser.add_argument('--momentum', type=float, default=0.9 , help='momentum')
parser.add_argument('--epochs', type=int, default=100,  help='number of epochs')
parser.add_argument('--lamda_base', type=float, default=5.0, help='特征蒸馏损失的权重因子 weight factor for feat dist')
parser.add_argument('--margin', type=float, default=0.5, help='边缘排名损失的边缘值 margin of loss margin')
parser.add_argument('--gamma', type=float, default=0.1, help='在每个里程碑处乘以学习率的因子 factor for multiply learning rate')
parser.add_argument('--knegatives', type=int, default=2, help='边缘排名损失的负样本数量 k negatives for loss margin')
parser.add_argument('--rehearsal', type=int, default=20,  help='每个类存储的样本数量 examplar stored per class')
parser.add_argument('--weight_decay', type=float, default=5e-4,help='优化器的权重衰减 weight decay')

parser.add_argument('--selection', type=str, default="herding",  help='样本选择方法的类型 type of selection of exemplar')
# herding主要目标是选择一组代表性样本，使得这些样本能够最好地代表整个数据集的分布，例如使得所选样本的特征均值尽量接近原始数据集的特征均值。

parser.add_argument("--exR", action="store_true", default=True, help="是否经验重放 experience replay")
parser.add_argument("--cosine", action="store_true", default=True, help="是否使用余弦分类器 cosine classifier")
parser.add_argument("--class_balance_finetuning", action="store_true", default=False, help="是否执行类平衡微调 class_balance_finetuning ")
parser.add_argument('--ft_epochs', default=20, type=int, help='类平衡微调的轮数 Epochs for class balance finetune')
parser.add_argument('--ft_base_lr', default=0.01, type=float,help='微调的基础学习率 Base learning rate for class balance finetune')
parser.add_argument('--ft_lr_strat', default=10, type=int, nargs='+', help='微调的学习率策略 Lr_strat for class balance finetune')
parser.add_argument("--less_forg", action="store_true", default=False, help="是否使用更少的遗忘损失less forgetting loss")
parser.add_argument("--ranking", action="store_true", default=False, help="是否使用边缘排名损失 loss margin ranking")

parser.add_argument("--list", nargs="+", default=["60", "100"], help="学习率调整的里程碑 Milestones for learning rate scale")

# 解析命令行参数
args = parser.parse_args()
torch.cuda.manual_seed(args.seed)

# 创建模型
model = create("cifar100", args.start, args.cosine)
model.cuda()

# 获取数据集
dataset_train, dataset_test = get_dataset(args.dataset)
train_transform, val_transform, test_transform = getTransform(args.dataset)

# 创建增量学习场景
scenario_train = ClassIncremental(dataset_train, increment=args.increment, initial_increment=args.start, transformations=[train_transform])
scenario_val = ClassIncremental(dataset_test, increment=args.increment, initial_increment=args.start, transformations=[val_transform])
assert scenario_train.nb_tasks == scenario_val.nb_tasks

# 打印类和任务数
print(f"Number of classes: {scenario_train.nb_classes}") # 数据集中样本类别数
print(f"Number of tasks: {scenario_val.nb_tasks}") # 根据初始类别数量和每次任务增加的类别数量计算任务数
# 例如，数据集默认100类，本次训练初始类别数量为50，每次任务增加的类别数量为50，则只有2个任务.

# 初始化样本集
exemplar_set = ExemplarSet(transform=train_transform)
scheduling = [int(args.list[0]) , int(args.list[1])]

# 定义损失函数
criterion_cls = nn.CrossEntropyLoss()
previous_net = None
accs = []
task_classes = []

# 任务循环
for task_id, train_taskset in enumerate(scenario_train):
    task_classes.extend(train_taskset.get_classes())
    if task_id > 0 and args.exR:
        train_taskset.add_samples(exemplar_set.data, exemplar_set.targets)
    print(f"TASK {task_id}")
    val_taskset = scenario_val[:task_id+1]
    sampler = get_sampler(train_taskset._y)
    val_loader = DataLoader(val_taskset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_lr = MultiStepLR(optimizer, milestones=scheduling, gamma=args.gamma) 
    train_loader = DataLoader(train_taskset, batch_size=args.batch_size, sampler=sampler)
     # 如果是第一个任务，lambda和旧类集为空
    if task_id == 0:
        lamda = None
        old_classes = None
    # 否则，根据当前和之前的类数量计算lambda，并获取旧类集
    else:
        lamda = args.lamda_base * np.sqrt(scenario_train[task_id].nb_classes / scenario_train[:task_id].nb_classes)
        old_classes = scenario_train[:task_id].get_classes()
    best_acc_on_task = 0

    # 训练和验证循环
    for epoch in range(args.epochs):
        train(args, train_loader, model, task_id, criterion_cls, previous_net, optimizer, epoch, lamda, old_classes)
        acc_val, loss_val = validate(model, val_loader)
        if acc_val > best_acc_on_task:
            best_acc_on_task = acc_val
            print(f"Saving best model\t ACC:{best_acc_on_task}")
            save_model(model, task_id)
        scheduler_lr.step()
        print(f"VALIDATION \t Epoch: {epoch}/{args.epochs}\t loss: {loss_val}\t acc: {acc_val}")
    
    # 加载最佳模型
    load_model(model, task_id)
    print("Loading best model...")
    model = load_model(model, task_id)
    model.cuda()

    # 样本选择和更新
    #if task_id < scenario_train.nb_tasks - 1:
    if args.exR:
        print(f"Selecting {args.rehearsal} exemplar per class from task {task_id}")
        for c in task_classes:
            images_in_c, labels_in_c = get_all_images_per_class(train_taskset, c)
            indexes = perform_selection(args, images_in_c, labels_in_c, model, val_transform)
            exemplar_set.update_data_memory(images_in_c[indexes], labels_in_c[indexes])
    
    # 类平衡微调
    if args.class_balance_finetuning and task_id > 0:
        print("Class Balance Finetuning")
        optimizer = optim.SGD(model.parameters(), lr=args.ft_base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler_lr = MultiStepLR(optimizer, milestones=[int(args.ft_lr_strat)], gamma=args.gamma)
        loader_balanced = DataLoader(exemplar_set, batch_size=args.batch_size, shuffle=True)
        best_acc_on_task = 0
        for epoch in range(args.ft_epochs):
            balanced_finetuning(args, loader_balanced, model, task_id, criterion_cls, optimizer,epoch)
            acc_val, loss_val = validate(model, val_loader)
            print(f"VALIDATION \t Epoch: {epoch}/{args.ft_epochs}\t loss: {loss_val}\t acc: {acc_val}")
            if acc_val > best_acc_on_task:
                best_acc_on_task = acc_val
            scheduler_lr.step()
        #print(f"VALIDATION \t Epoch: {epoch}/{args.epochs}\t loss: {loss_val}\t acc: {acc_val}")
        accs.append(best_acc_on_task)
        #print(f"ACCURACY \t  {acc_val}")
    else:
        accs.append(best_acc_on_task)
    
    # 扩展模型以适应新任务
    if task_id < scenario_train.nb_tasks - 1 :  
        #print("Expanding")
        previous_net = copy.deepcopy(model)         
        model.expand_classes(scenario_train[task_id+1].nb_classes)
        model.cuda()
        task_classes = []

# 打印每个任务的准确率和平均准确率
print(accs)
print(np.mean(np.array(accs)))