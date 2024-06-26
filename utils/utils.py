import torch
import numpy as np
from torchvision import datasets, models, transforms
from continuum.datasets.pytorch import CIFAR100
from continuum.datasets.imagenet import ImageNet100
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler

# 计算给定输出和目标的 top-k 准确率
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # 获取 top-k 的预测结果
        pred = pred.t()  # 转置预测结果
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 计算预测结果是否正确

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # 计算正确预测的数量
            res.append(correct_k.mul_(100.0 / batch_size))  # 计算并存储准确率
        return res

# 带有异常处理的迭代器
def next_with_clause(iterator, data_loader):
    try:
        x, y = next(iterator)
    except Exception as e:
        iterator = iter(data_loader)
        x, y = next(iterator)
    return x, y, iterator

# 从内存中提取一个批次的数据
def extract_batch_from_memory(memory_iterator, memory_data_loader, batch_size):
    """
    Extract a batch from memory
    """
    memory_imgs, memory_labels, memory_iterator = next_with_clause(memory_iterator, memory_data_loader)

    assert memory_labels.shape[0] == memory_imgs.shape[0]
    while memory_labels.shape[0] < batch_size:
        m, l, memory_iterator = next_with_clause(memory_iterator, memory_data_loader)
        memory_imgs = torch.cat((memory_imgs, m))
        memory_labels = torch.cat((memory_labels, l))
        memory_iterator = iter(memory_data_loader)

    memory_imgs = memory_imgs[:batch_size]
    memory_labels = memory_labels[:batch_size]

    return memory_imgs, memory_labels, memory_iterator

# 获取每个类的所有图像
def get_all_images_per_class(task_set, target):
    indexes = np.arange(len(task_set))
    images, labels, task = task_set.get_raw_samples(indexes)

    images_of_class = images[labels == target]
    labels_of_class = labels[labels == target]

    return images_of_class, labels_of_class

# 根据数据集名称获取数据变换
def getTransform(dataset):
    if dataset == "cifar100":
        train_transform = transforms([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        val_transform = transforms([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        test_transform = transforms([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])

    else:
        raise "Dataset not recognized"
    return train_transform, val_transform, test_transform

# 获取数据集
def get_dataset(dataset):
    if dataset == "cifar100":
        dataset_train = CIFAR100("data", download=True, train=True)
        dataset_test = CIFAR100("data", download=True, train=False)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    print(f"{dataset} dataset_train: {type(dataset_train)}, dataset_test: {type(dataset_test)}")
    return dataset_train, dataset_test

# 创建加权随机采样器
def get_sampler(target):
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

# 保存模型
def save_model(net, task_id):
    net_state_dict = net.state_dict()
    checkpoint ={
        'net_state_dict': net_state_dict,
                }
    ckpt_path = f"ckpt_task_{task_id}.pt" 
    torch.save(checkpoint, ckpt_path)

# 加载模型
def load_model(net, task_id):
    ckpt_path = f"ckpt_task_{task_id}.pt"
    net.load_state_dict(torch.load(ckpt_path)['net_state_dict'])
    return net
