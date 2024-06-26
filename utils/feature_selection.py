import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from validate import extract_features
from utils.ExemplarSet import ExemplarSet as ImagesDataset

# 定义选择示例集的方法
def perform_selection(args, images, labels, net, transform):
    if args.selection == "random":
        # 随机选择示例
        indexes = np.random.permutation(len(images))[:args.rehearsal]
    else:
        # 使用给定的变换构建数据集
        dataset_class_c = ImagesDataset(images, labels, transform)
        # 使用 DataLoader 加载数据
        loader = DataLoader(dataset_class_c, batch_size=args.batch_size, shuffle=False, drop_last=False)
        # 提取特征
        features = extract_features(args, net, loader)
        if args.selection == "closest":
            # 选择与均值最近的示例
            indexes = closest_to_mean(features, args.rehearsal)
        elif args.selection == "herding":
            # 使用 iCaRL 选择方法选择示例
            indexes = icarl_selection(features, args.rehearsal)
    return indexes

# 计算 L2 距离的函数
def _l2_distance(x, y):
    return np.power(x - y, 2).sum(-1)

# iCaRL 选择方法——herding
def icarl_selection(features, nb_examplars):
    D = features.T  # 转置特征矩阵
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)  # 对特征进行归一化
    mu = np.mean(D, axis=1)  # 计算特征的均值
    herding_matrix = np.zeros((features.shape[0],))  # 初始化 herding 矩阵

    w_t = mu  # 初始化 w_t 为均值向量
    iter_herding, iter_herding_eff = 0, 0  # 初始化迭代计数器

    while not (np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)  # 计算 w_t 与特征矩阵的点积
        ind_max = np.argmax(tmp_t)  # 找到最大值的索引
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding  # 更新 herding 矩阵
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]  # 更新 w_t

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000  # 将未选择的索引设为 10000

    return herding_matrix.argsort()[:nb_examplars]  # 返回选择的索引

# 选择与均值最近的示例
def closest_to_mean(features, nb_examplars):
    # features = features / (np.linalg.norm(features, axis=0) + 1e-8)
    class_mean = np.mean(features, axis=0)  # 计算特征均值

    return _l2_distance(features, class_mean).argsort()[:nb_examplars]  # 返回与均值最近的示例索引
