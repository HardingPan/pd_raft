import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)  # 对两图特征使用矩阵乘法得到相关性查找表

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)  # (b,h,w,1,h,w) -> (bhw,1,h,w)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)  # 使用平均 pooling 的方式获得多尺度查找表
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # (b,2,h,w) -> (b,h,w,2) 当前坐标，包含x和y两个方向，由 meshgrid() 函数得到.
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]  # (bhw,1,h,w) 某一尺度的相关性查找表
            # torch.linspace(start, end, steps)  start:开始值 end:结束值 steps:分割的点数,默认为100
            dx = torch.linspace(-r, r, 2*r+1)  # (2r+1) x方向的相对位置查找范围
            dy = torch.linspace(-r, r, 2*r+1)  # (2r+1) y方向的相对位置查找范围
            # 相对坐标
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)  # 查找窗 (2r+1,2r+1,2)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i  # (b,h,w,2) -> (bhw,1,1,2) 某一尺度下的坐标
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)  # (2r+1,2r+1,2) -> (1,2r+1,2r+1,2) 查找窗
            coords_lvl = centroid_lvl + delta_lvl  # (bhw,1,1,2) + (1,2r+1,2r+1,2) -> (bhw,2r+1,2r+1,2) 可以形象理解为：对于 bhw 这么多待查找的点，每一个点需要搜索 (2r+1)*(2r+1) 邻域范围内的其他点，每个点包含 x 和 y 两个坐标值

            corr = bilinear_sampler(corr, coords_lvl)  # (bhw,1,2r+1,2r+1) 在查找表上搜索每个点的邻域特征，获得相关性图
            corr = corr.view(batch, h1, w1, -1) # (bhw,1,2r+1,2r+1) -> (b,h,w,(2r+1)*(2r+1))
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
    # 得到correlation volume
    def corr(fmap1, fmap2):
        # fmap1.shape: (b, c, h, w)
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)  # 第一帧图特征 (b,c,h,w) -> (b,c,hw)
        fmap2 = fmap2.view(batch, dim, ht*wd)  # 第二帧图特征 (b,c,h,w) -> (b,c,hw)

        corr = torch.matmul(fmap1.transpose(1,2), fmap2)  # (b,hw,c) * (b,c,hw) -> (b,hw,hw) 后两维使用矩阵乘法，第一维由广播得到
        corr = corr.view(batch, ht, wd, 1, ht, wd)  # (b,hw,hw) -> (b,h,w,1,h,w)
        return corr / torch.sqrt(torch.tensor(dim).float())  # 这里除的意义不是很明确，应该有一定的数学意义，有了解的小伙伴可以在评论区补充一下（不影响理解）

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)  # (bhw,2r+1,2r+1,1)
    xgrid = 2*xgrid/(W-1) - 1  # x方向归一化
    ygrid = 2*ygrid/(H-1) - 1  # y方向归一化

    grid = torch.cat([xgrid, ygrid], dim=-1)  # (bhw,2r+1,2r+1,2)
    img = F.grid_sample(img, grid, align_corners=True)  # img: (bhw,1,h,w) -> (bhw,1,2r+1,2r+1) 根据搜索范围 grid 在查找表 img 中采样对应特征

    return img
