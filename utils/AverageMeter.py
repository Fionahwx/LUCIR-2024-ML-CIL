# 用于在训练和评估模型时跟踪并计算某些度量（如损失、精度等）的平均值。
# 它提供了一个简单的方法来更新和获取这些度量的当前值、总和、计数和平均值。

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 假设当前有 N 个值，其中总和为 ∑ (i=1~N) xi, 平均值为 avg。
# 新增一个值 x，数量为 n。
# 更新总和为：new_sum = old_sum + x * n
# 更新计数为：new_count= old_count + n
# 更新平均值为：new_avg= new_count / new_sum


