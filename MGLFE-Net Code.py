from typing import Tuple
from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PCCA import PCCA
from models.RGSA import RGSA
from models.CSSA import CSSA
from models.timm.layers import DropPath, to_2tuple, LayerNorm2d
from models.Biformer import biformer_base

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class main_model(nn.Module):
    def __init__(self, num_classes, in_chans=3, embed_dim=96, HFF_dp=0., drop_rate=0.,
                 conv_depths=(2, 2, 2, 2), conv_dims=(96, 192, 384, 768), conv_drop_path_rate=0.,
                 conv_head_init_scale: float = 1., **kwargs):
        super().__init__()

        ###### Local Branch Setting #######

        self.downsample_layers = nn.ModuleList()  # stem + 3 stage downsample
        stem = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),
                             LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # stage2-4 downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(conv_dims[i], conv_dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages_conv = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))]
        cur = 0

        # Build stacks of blocks in each stage
        for i in range(4):
            stage = nn.Sequential(
                *[Local_block(dim=conv_dims[i], drop_rate=dp_rates[cur + j])
                  for j in range(conv_depths[i])]
            )
            self.stages_conv.append((stage))
            cur += conv_depths[i]

        self.conv_norm = nn.LayerNorm(conv_dims[-1], eps=1e-6)  # final norm layer
        self.conv_head = nn.Linear(conv_dims[-1], num_classes)
        self.conv_head.weight.data.mul_(conv_head_init_scale)
        self.conv_head.bias.data.mul_(conv_head_init_scale)

        ###### Global Branch Setting ######
        self.bi_model = biformer_base(False)
        # split image into non-overlapping patches
        self.patch_embed = self.patch_embed = StemLayer(in_chs=in_chans,
                                                        out_chs=embed_dim, stride=4, norm_layer=LayerNorm2d)
        self.pos_drop = nn.Dropout(p=drop_rate)
        ###### Hierachical Feature Fusion Block Setting #######

        self.fu1 = Fusion(channel=96,drop_rate=0.)
        self.fu2 = Fusion(channel=192,drop_rate=0.)
        self.fu3 = Fusion(channel=384,drop_rate=0.)
        self.fu4 = Fusion(channel=768,drop_rate=0.)
        self.enhance = CSSA(768)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, imgs):

        ######  Global Branch ######
        x_s = self.patch_embed(imgs)
        x_s = self.pos_drop(x_s)
        x_s_1 = self.bi_model.stages[0](x_s)
        x_s1 = self.bi_model.downsample_layers[1](x_s_1)
        x_s_2 = self.bi_model.stages[1](x_s1)
        x_s2 = self.bi_model.downsample_layers[2](x_s_2)
        x_s_3 = self.bi_model.stages[2](x_s2)
        x_s3 = self.bi_model.downsample_layers[3](x_s_3)
        x_s_4 = self.bi_model.stages[3](x_s3)

        ######  Local Branch ######
        x_c = self.downsample_layers[0](imgs)
        x_c_1 = self.stages_conv[0](x_c)
        x_c = self.downsample_layers[1](x_c_1)
        x_c_2 = self.stages_conv[1](x_c)
        x_c = self.downsample_layers[2](x_c_2)
        x_c_3 = self.stages_conv[2](x_c)
        x_c = self.downsample_layers[3](x_c_3)
        x_c_4 = self.stages_conv[3](x_c)

        ###### Hierachical Feature Fusion Path ######
        x_f_1 = self.fu1(x_c_1, x_s_1, None)
        x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1)
        x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
        x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3)
        after_cssa = self.enhance(x_f_4)
        x_fu = self.conv_norm(after_cssa.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        x_fu = self.conv_head(x_fu)

        return x_fu
    
##### Local Feature Block Component #####
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Local_block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_rate=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class Fusion(nn.Module):
    def __init__(self, channel,drop_rate=0.):
        super(Fusion, self).__init__()
        self.W_l = Conv(channel, channel, 1, bn=True, relu=False)
        self.W_g = Conv(channel, channel, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.norm1 = LayerNorm(channel*3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(channel * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(channel + channel + channel, eps=1e-6, data_format="channels_first")
        self.Updim = Conv(channel // 2, channel, 1, bn=True, relu=True)
        self.W3 = Conv(channel * 3, channel, 1, bn=True, relu=False)
        self.W = Conv(channel * 2, channel, 1, bn=True, relu=False)
        self.gelu = nn.GELU()
        self.residual = IRMLP(channel * 3, channel)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.PCCA = PCCA(kernel_size=3,inchanel=channel,outchanel=channel)
        self.RGSA = RGSA(channel)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, l, g, f):
        W_local = self.W_l(l)  # local feature from Local Feature Block
        W_global = self.W_g(g)  # global feature from Global Feature Block
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            f = torch.cat([W_f, W_local, W_global], 1)
            f = self.norm1(f)
            f = self.W3(f)
            f = self.gelu(f)
        else:
            f = torch.cat([W_local, W_global], 1)
            f = self.norm2(f)
            f = self.W(f)
            f = self.gelu(f)

        # spatial attention for ConvNeXt branch
        l = self.RGSA(l)

        # channel attetion for transformer branch
        g = self.PCCA(g)

        fuse = torch.cat([g, l, f], 1)
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = self.drop_path(fuse)
        return fuse

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


class StemLayer(nn.Module):
    """ Size-agnostic implementation of 2D image to patch embedding,
        allowing input size to be adjusted during model forward operation
    """

    def __init__(
            self,
            in_chs=3,
            out_chs=96,
            stride=4,
            norm_layer=LayerNorm2d,
    ):
        super().__init__()
        stride = to_2tuple(stride)
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        assert stride[0] == 4  # only setup for stride==4
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=7,
            stride=stride,
            padding=3,
        )
        self.norm = norm_layer(out_chs)

    def forward(self, x):
        B, C, H, W = x.shape  # 确保 x 是四维张量 (Batch, Channels, Height, Width)

        # 宽度方向填充
        pad_w = (self.stride[1] - W % self.stride[1]) % self.stride[1]
        x = F.pad(x, (0, pad_w))  # 右侧填充

        # 高度方向填充
        pad_h = (self.stride[0] - H % self.stride[0]) % self.stride[0]
        x = F.pad(x, (0, 0, 0, pad_h))  # 底部填充

        x = self.conv(x)
        x = self.norm(x)
        return x


def MLGFE_Net(num_classes: int):
    model = main_model(conv_depths=(4, 4, 18, 2),
                       num_classes=num_classes)
    return model

