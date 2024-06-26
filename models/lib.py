import torch
import torch.nn as nn
from torch.autograd import Function

# 定义 Weldon 池化层
class WeldonPool2d(nn.Module):

    def __init__(self, kmax=1, kmin=None, **kwargs):
        super(WeldonPool2d, self).__init__()
        self.kmax = kmax  # 定义最大池化的实例数
        self.kmin = kmin  # 定义最小池化的实例数
        if self.kmin is None:
            self.kmin = self.kmax  # 如果 kmin 未指定，则默认与 kmax 相同

        print("Using Weldon Pooling with kmax={}, kmin={}.".format(self.kmax, self.kmin))
        self._pool_func = self._define_function()  # 定义池化操作的函数

    def forward(self, input):
        return self._pool_func(input)  # 前向传播调用池化函数

    def _define_function(self):
        # 定义 Weldon 池化操作的函数
        class WeldonPool2dFunction(Function):
            @staticmethod
            def get_number_of_instances(k, n):
                if k <= 0:
                    return 0
                elif k < 1:
                    return round(k * n)
                elif k > n:
                    return int(n)
                else:
                    return int(k)

            @staticmethod
            def forward(ctx, input):
                # 获取批次信息
                batch_size = input.size(0)
                num_channels = input.size(1)
                h = input.size(2)
                w = input.size(3)

                # 获取区域数量
                n = h * w

                # 获取最大和最小实例的数量
                kmax = WeldonPool2dFunction.get_number_of_instances(self.kmax, n)
                kmin = WeldonPool2dFunction.get_number_of_instances(self.kmin, n)

                # 对分数进行排序
                sorted, indices = input.new(), input.new().long()
                torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True, out=(sorted, indices))

                # 计算最大实例的分数
                indices_max = indices.narrow(2, 0, kmax)
                output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

                if kmin > 0:
                    # 计算最小实例的分数
                    indices_min = indices.narrow(2, n-kmin, kmin)
                    output.add_(sorted.narrow(2, n-kmin, kmin).sum(2).div_(kmin)).div_(2)

                # 保存输入用于反向传播
                ctx.save_for_backward(indices_max, indices_min, input)

                # 返回正确尺寸的输出
                return output.view(batch_size, num_channels)

            @staticmethod
            def backward(ctx, grad_output):

                # 获取输入
                indices_max, indices_min, input, = ctx.saved_tensors

                # 获取批次信息
                batch_size = input.size(0)
                num_channels = input.size(1)
                h = input.size(2)
                w = input.size(3)

                # 获取区域数量
                n = h * w

                # 获取最大和最小实例的数量
                kmax = WeldonPool2dFunction.get_number_of_instances(self.kmax, n)
                kmin = WeldonPool2dFunction.get_number_of_instances(self.kmin, n)

                # 计算最大实例的梯度
                grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmax)
                grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, indices_max, grad_output_max).div_(kmax)

                if kmin > 0:
                    # 计算最小实例的梯度
                    grad_output_min = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmin)
                    grad_input_min = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, indices_min, grad_output_min).div_(kmin)
                    grad_input.add_(grad_input_min).div_(2)

                return grad_input.view(batch_size, num_channels, h, w)

        return WeldonPool2dFunction.apply

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax
                                                        ) + ', kmin=' + str(self.kmin) + ')'
