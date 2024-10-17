
import torch
import torch.nn as nn

from torch.nn import functional as F
import numpy as np  

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,padding, dilation, groups, bias)

        if kernel_size == 1:
            self.ind = True
        else:
            self.ind = False            
            self.out_channels = out_channels
            self.ks = kernel_size

            ws = kernel_size
            # 自适应的平均池化，指定输出大小为 3*3 就是 9
            self.avg_pool = nn.AdaptiveAvgPool1d(ws*ws)
            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)         # num_lat = 5

            self.ce = nn.Linear(ws*ws, self.num_lat, False)         # 输入特征数为9，输出特征数为5
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)

            self.act = nn.ReLU(inplace=True)
            
            # // 表示返回不大于除法结果的一个最大整数，除了第一次输入通道数为2以外，其余输入通道数都为16
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels

            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)     # 输入特征数 g，输出特征数16
            self.ci_bn = nn.BatchNorm1d(out_channels)

            self.gd = nn.Linear(self.num_lat, kernel_size * kernel_size, False)     # 输入特征数为5，输出特征数为9
            self.gd2 = nn.Linear(self.num_lat, kernel_size * kernel_size, False)    # 输入特征数为5，输出特征数为9

            # Unfold根据卷积核的大小，将图片切割成不同的patch
            # Unfold的输出结果：(bs, in_channel*ks*ks, 块数)
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
            self.sig = nn.Sigmoid()



    def forward(self, x, y):
        # 如果是 1*1 卷积，那就直接调用库函数
        if self.ind:
            return F.conv2d(x, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()       # x : (4,2,358,358)
            weight = self.weight        # weight : (16,2,3,3)

            gl = self.avg_pool(y.transpose(1,2)).view(b,c,-1)       # gl : (batch_size ,in_channel,9)

            # 经过一个线性层
            out = self.ce(gl)       # out : (batch_size ,in_channel,5)
            ce2 = out           # ce2 : (batch_size ,in_channel,5)
            # 经过一个 BatchNorm1d
            out = self.ce_bn(out)       # out : (batch_size ,in_channel,5)
            # 经过一个ReLU激活
            out = self.act(out)         # out : (batch_size ,in_channel,5)
            # 再经过一个线性层
            out = self.gd(out)          # out : (batch_size ,in_channel,9)

            # 将原始的out经过不同的线性层
            if self.g >3:
                oc = self.ci(self.act(self.ci_bn2(ce2).view(b, c//self.g, self.g, -1).transpose(2,3))).transpose(2,3).contiguous()
            else:
                # contiguous就是断开赋值前后的联系，相当于深拷贝
                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2,1))).transpose(2,1).contiguous()         # oc : (4,16,5)

            oc = oc.view(b,self.out_channels,-1)      # oc : (batch_size ,out_channels ,5)
            oc = self.ci_bn(oc)         # oc : (batch_size ,out_channels ,5)
            oc = self.act(oc)           # oc : (batch_size ,out_channels ,5)
            oc = self.gd2(oc)           # oc : (batch_size ,out_channels ,9)
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.out_channels, 1, self.ks, self.ks))      # out : (batch_size ,out_channels ,2,3,3)

            x_un = self.unfold(x)           # x_un : (batch_size ,18,128164)
            b, _, l = x_un.size()   
            out = (out * weight.unsqueeze(0)).view(b, self.out_channels, -1)      # out : (batch_size ,16,18)
           
            # 用矩阵乘法代替卷积操作
            return torch.matmul(out, x_un).view(b, self.out_channels, int(np.sqrt(l)), int(np.sqrt(l)))       # (batch_size , out_channel,358,358)
            
