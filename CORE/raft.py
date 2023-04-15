import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CORE.extraction import BasicEncoder#, SmallEncoder#feature network, context network
from CORE.correlation import CorrBlock, AlternateCorrBlock#相关性查找表
from CORE.update import BasicUpdateBlock#, SmallUpdateBlock#ConVGRU
from CORE.utils.utils import bilinear_sampler, coords_grid, upflow8#计算flow、上采样
#
import matplotlib.pyplot as plt
from CORE.utils import flow_viz
import cv2
#https://www.cnblogs.com/jimchen1218/p/14315008.html
try:
    #利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

#①初始化参数
        """
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        """
        #这里选用4.8M的RAFT框架
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        #相关体数
        args.corr_levels = 4
        args.corr_radius = 4
        #dropout值
        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

#②子网络架构
        # feature network, context network, and update block
        """
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        """
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm2d):#如果m的类型与nn.BatchNorm2d的类型相同则返回 True，否则返回 False
    #             m.eval()

#③初始化光流f0
    #feature encoder network outputs:features at 1/8 resolution(R^H*W*3->G^H/8*W/8*256)
    # Flow is represented as difference between two coordinate grids flow = coords1 - coords0
    def initialize_flow(self, img):
        N, C, H, W = img.shape#这里cam[0]:torch.Size([1, 3, 160, 320])
        coords0 = coords_grid(N, H//8, W//8).to(img.device)# (1，1,20,40)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)# (1，1,20,40)
        return coords0, coords1

#④上采样：mask * up_flow
    #Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination
    #提取每个像素点以及周围的 8 邻域像素点特征（总共 9 个像素点）重新排列到 channel 维度上
    def upsample_flow(self, flow, mask):
        #1)mask
        N, _, H, W = flow.shape#这里torch.Size([1, 2, 20, 40])
        mask = mask.view(N, 1, 9, 8, 8, H, W) # (N,9*8*8,20,40) -> (N,1,9,8,8,20,40)
        mask = torch.softmax(mask, dim=2)  # 权重归一化
        #2)up_flow
        #8*flow:上采样后图像的尺度变大了，为了匹配尺度增大的像素坐标，光流(flow=coords1 - coords0)也要按同样的倍率（8 倍）上采样
        up_flow = F.unfold(8 * flow, [3,3], padding=1) #每一列的元素为滑动窗口(只卷不积)依次所覆盖的内容 (b,2,h,w) -> (b,2*3*3,h*w)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W) # (b,2*3*3,h*w) -> (b,2,9,1,1,h,w)
        up_flow = torch.sum(mask * up_flow, dim=2) # (b,1,9,8,8,h,w) * (b,2,9,1,1,h,w) -> (b,2,9,8,8,h,w) ->(sum) (b,2,8,8,h,w)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # (b,2,8,8,h,w) -> (b,2,h,8,w,8)
        return up_flow.reshape(N, 2, 8*H, 8*W)  # (b,2,h,8,w,8) -> (b,2,8h,8w)




    ##Estimate optical flow between pair of frames
    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):

        # step 1：网络输入-图像预处理
        image1 = 2 * (image1 / 255.0) - 1.0#图像归一化
        image2 = 2 * (image2 / 255.0) - 1.0#图像归一化
        image1 = image1.contiguous()#类似深拷贝；调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系
        image2 = image2.contiguous()
        hdim = self.hidden_dim#这里128
        cdim = self.context_dim#这里128

        #step 2：Feature Encoder 提取两图特征（权值共享）
        with autocast():
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # step 3：初始化相关性查找表时，调用 __init__() 函数；
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)#这里self.args.corr_radius=4
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # step 4：Context Encoder提取第一帧图特征
        #GRU输入特征之一：net为GRU的隐状态，inp后续与其他特征结合作为 GRU 的一般输入
        with autocast():
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # step 5：更新光流
        # 初始化光流的坐标信息
        # coords0 为初始时刻的坐标，coords1 为当前迭代的坐标
        coords0, coords1 = self.initialize_flow(image1)#此处两坐标数值相等
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = [] #key：对每次迭代的光流都进行了上采样
        for itr in range(iters):
            #1）初始的coords1
            coords1 = coords1.detach()#key:使之切断反向传播，不具有更新属性
            #2）更新的coords1（coords0做基准一直不变）
            corr = corr_fn(coords1) #从相关性查找表中获取当前坐标的对应特征（查找对应特征时，调用 __call__() 函数）
            flow = coords1 - coords0

            #a）self.update_block的输入：context网络的输出（net, inp）-帧1信息；从相关性查找表中获取各坐标的对应特征-帧1、2信息；flow-包含抽象更新趋势delta_flow
            with autocast():
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)# 计算当前迭代的光流
            #b）F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow#更新光流

            #3）upsample predictions（目的：训练网络）
            # step 6：上采样光流（此处为了训练网络，对每次迭代的光流都进行了上采样，实际 inference 时，只需要保留最后一次迭代后的上采样）
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
        return flow_predictions

