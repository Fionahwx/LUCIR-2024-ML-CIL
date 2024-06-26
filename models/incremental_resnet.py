import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.resnet_cifar import resnet32
from models.CosineClassifier import CosineClassifier
from models.my_resnet import resnet_rebuffi

# 定义一个增量学习的ResNet模型
class Incremental_ResNet(nn.Module):
    def __init__(self, backbone="resnet32", starting_classes=10, cosine=True):  # 初始化，默认使用resnet32作为backbone，初始类数为10，使用余弦分类器
        super(Incremental_ResNet, self).__init__()
        self.cosine = cosine
        self.backbone = resnet_rebuffi()  # 使用自定义的resnet_rebuffi作为backbone
        # if backbone == "resnet32":
        #     self.backbone = resnet32(num_classes=starting_classes)
        # elif backbone =="resnet18":
        #     self.backbone = resnet18(pretrained=False, num_classes=starting_classes)
        self.feat_size = self.backbone.out_dim  # 获取backbone输出的特征维度
        #self.fc1 = nn.Linear(self.feat_size, starting_classes, bias=not(self.cosine))
        if self.cosine:
            self.fc1 = CosineClassifier(self.feat_size, starting_classes)  # 使用余弦分类器
        else:
            self.fc1 = nn.Linear(self.feat_size, starting_classes)  # 使用普通的全连接层分类器

    def forward(self, x):
        x = self.backbone(x)  # 获取特征

        y = self.fc1(x)  # 进行分类

        return x, y  # 返回特征和分类结果

    def expand_classes(self, new_classes):
        # 扩展类数
        old_classes = self.fc1.weight.data.shape[0]  # 获取旧的类数
        old_weight = self.fc1.weight.data  # 保存旧的权重
        if self.cosine:
            self.fc1 = CosineClassifier(self.feat_size, old_classes + new_classes)  # 创建新的余弦分类器
        else:
            self.fc1 = nn.Linear(self.feat_size, old_classes + new_classes)  # 创建新的全连接层分类器
        self.fc1.weight.data[:old_classes] = old_weight  # 保留旧的权重

    def classify(self, x):
        # 进行分类
        y = self.fc1(x)
        return x, y

# 创建ResNet32增量模型
def ResNet32Incremental(starting_classes=10, cosine=True):
    model = Incremental_ResNet(backbone="resnet32", starting_classes=starting_classes, cosine=cosine)
    return model

# 模型工厂字典
__factory = {
    'cifar100': "resnet32"
}

# 创建指定数据集的模型
def create(dataset, classes, cosine):
    if dataset not in __factory.keys():
        raise KeyError(f"Unknown Model: {dataset}")  # 如果数据集不在工厂字典中，抛出错误
    return Incremental_ResNet(__factory[dataset], starting_classes=classes, cosine=cosine)
