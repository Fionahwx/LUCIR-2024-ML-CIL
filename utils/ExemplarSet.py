import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
from PIL import Image

# 定义 ExemplarSet 类，继承自 PyTorch 的 Dataset 类
# 用于管理和处理不断变化的数据集，特别是在增量学习或在线学习的场景中，通过调用 update_data_memory 方法可以动态地添加新的数据和标签。

class ExemplarSet(Dataset):
    # 初始化方法
    def __init__(self, data=None, targets=None, transform=None):
        self.data = data  # 数据集
        self.targets = targets  # 数据集对应的标签
        # 如果有传入变换，则使用 transforms.Compose 将变换组合起来
        self.transform = None if transform is None else transforms.Compose(transform)

    # 获取数据集中的一个元素
    def __getitem__(self, index):
        x = self.data[index]  # 获取第 index 个数据
        y = self.targets[index]  # 获取第 index 个数据对应的标签

        # 如果定义了变换
        if self.transform:
            # 将数据转换为 PIL Image 类型，并进行变换
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)

        return x, y  # 返回数据和标签

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)

    # 更新数据集内存
    def update_data_memory(self, new_data, new_targets):
        # 确保新数据和新标签的数量相同
        assert new_data.shape[0] == new_targets.shape[0]
        # 如果当前数据为空，则直接赋值新数据，否则将新数据添加到现有数据后面
        self.data = new_data if self.data is None else np.concatenate((self.data, new_data))
        # 同样地，更新标签
        self.targets = new_targets if self.targets is None else np.concatenate((self.targets, new_targets))
