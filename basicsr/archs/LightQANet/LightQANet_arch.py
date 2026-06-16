import torch
import torch.nn.functional as F
from torch import nn as nn
import copy
import numpy as np
import math
import sys
import numbers
try:
    import clip
except ImportError:
    clip = None
import matplotlib.pyplot as plt
import os


from einops import rearrange
from einops.layers.torch import Rearrange
import sys

from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.network_swinir import RSTB
from basicsr.archs.ridcp_utils import ResBlock, CombineQuantBlock
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.archs.CrossAttention.attention_processor import CrossAttention_wx2 as Cross_Attention


# --------------------------------------------------------
# 2024.04.16

# encoder
# 将 prompt模块作为一个并联的模块使用
#   利用 prompt 处理encoder block的输出特征，得到 prompt feat
#   将 prompt feat 和 encoder block 输出特征融合
#   然后送到 transformer中处理

# Fuse_aft_block：
#   1 浅层的跳跃连接，控制纹理
#   2 light prompt: 利用 dec_feat + residual 对 prompt 进行加权，得到 prompt feat
#       融合方式：将 prompt feat 和 encoder block 输出特征融合
#       用 CNN 处理融合特征

# 实际上就是 v2_4的代码，已经跑起来了，就当做事v2.4

# 引入 domain invariant loss
# --------------------------------------------------------

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward2(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(FeedForward2, self).__init__()

        # hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Linear(dim, dim * ffn_expansion_factor, bias=bias)
        # self.hidden_fc = nn.Linear(hidden_features * 2, hidden_features * 2, bias=False)
        self.project_out = nn.Linear(dim * ffn_expansion_factor, dim, bias=bias)


    def forward(self, x):
        x = F.relu(self.project_in(x))
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Transformer Block，cross attention
class CrossTransformerBlock(nn.Module):
    def __init__(self, n_head, d_input, d_k, d_v, dropout=0.1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(CrossTransformerBlock, self).__init__()

        self.clip_dim_convert = nn.Linear(512, d_input)       # clip 的图像特征输出维度是512

        self.norm1 = nn.LayerNorm(d_input)
        self.norm12 = nn.LayerNorm(d_input)
        self.attn = Cross_Attention(n_head=n_head, d_input=d_input, d_k=d_input, d_v=d_input)
        self.norm2 = nn.LayerNorm(d_input)
        self.ffn = FeedForward2(d_input, ffn_expansion_factor, bias)

    def forward(self, input):
        # input: [x, condition, enc_feat]
        x = input[0]
        condition = input[1]

        x = self.norm1(x)
        condition = self.norm12(condition)

        x = x + self.attn(x, condition, condition)
        x = x + self.ffn(self.norm2(x))
        # x = x.view(batch_size, height, width, num_channels).permute(0, 3, 1, 2)

        return [x, condition]

def _visual_prompt(input):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import time

    t = time.time()
    b, c, h, w = input.shape
    num_prompts, channels = b, c
    in_np = input.cpu().detach().numpy()
    # 将每个 prompt 中的每个通道展平为一个 1024 维向量，总共 5*256 = 1280 个数据点
    data_flat = in_np.reshape(num_prompts * channels, -1)

    # 构造标签，每个通道对应其所属的 prompt（标签 0~4）
    labels = np.repeat(np.arange(num_prompts), channels)

    # 使用 t-SNE 将 1024 维数据降到 2 维，设置 perplexity 为 30
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    data_tsne = tsne.fit_transform(data_flat)

    # 绘制 t-SNE 的散点图，不同颜色代表不同的 prompt
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar(sc, ticks=range(num_prompts))
    plt.title("t-SNE of prompts")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(f'/data/data0/wuxu/codes/low_light/PromptEnhance/visual_net/visual_prompt/t_sne_prompt_{t}.png')
    plt.show()
##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    # prompt_dim: prompt 的维度
    # prompt_len: prompt 的个数，这里对应的应该是退化模式(还原任务)的个数：denoise、derain、defoggy...
    # prompt_size: 这是啥？
    # lin_dim: 输入特征映射的目标维度
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))   # prompt 可学习参数
        self.linear_layer = nn.Linear(lin_dim,prompt_len)       
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        
    def forward(self,x, test=False):
        # 1 GAP：x 经过 全局平均池化处理
        # 2 全连接：将GAP的结果映射到prompt的空间
        # 3 softmax：处理全连接的结果，得到 x 最终的映射结果，并将其与 prompt components 相乘

        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))       # global average pooling 操作

        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)       # 将特征映射到 权重空间        
        # t = self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        # t2 = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        ## 手动指定 weights
        # if test:
        #     define_prompt_weight = np.array([[0, 0, -100, 0, 0]])
        #     define_prompt_weight = torch.tensor(define_prompt_weight).cuda()
        #     # prompt_weights = define_prompt_weight
        #     print(f'prompt_weights: {prompt_weights}')
        #     print(f'define_prompt_weight: {define_prompt_weight}')

        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        # _visual_prompt(self.prompt_param.squeeze(0))
        self.prompt = prompt
        self.prompt_weights = prompt_weights
        
        # print(f'p: {prompt.shape}')
        return prompt


##########################################################################


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class Fuse_aft_block(nn.Module):

    '''
    跳跃连接的特征融合
    -1 cat decoder 、 encoder特征
    -2 根据cat的特征，获得 scale,shift 两个参数
    -3 y = a*b + c的形式获得融合特征： y = scale * dec feat + shift

    引入prompt
    '''

    def __init__(self, in_ch,
        out_ch,
        embed_dim=128,
        num_heads=4,
        num_blocks=None,
        region_size=5,
        prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 96,):

        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)
        self.encode_enc2 = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.prompt_light = PromptGenBlock(prompt_dim=prompt_dim,prompt_len=prompt_len,prompt_size = prompt_size,lin_dim = lin_dim)
        self.reduce_channel = nn.Conv2d(out_ch * 2, out_ch,kernel_size=1,bias=False)
        self.fuse = ResBlock(out_ch, out_ch)

        # self.prompt = PromptGenBlock(prompt_dim=prompt_dim,prompt_len=prompt_len,prompt_size = prompt_size,lin_dim = lin_dim)
        # self.reduce_channel2 = nn.Conv2d(out_ch * 2, out_ch,kernel_size=1,bias=False)
        # self.fuse2 = ResBlock(out_ch, out_ch)


    def forward(self, enc_feat, style_feat=None, dec_feat=None, w1=1, w2=1, w3=1, test=False):
        '''

        :param enc_feat: low level feats,弥补纹理信息
        :param dec_feat: decoder feats
        :param style_feat: it is used for computing variance，用于控制输出图像的对比度、亮度（style feats）
        :param w1: 控制纹理信息的影响
        :param w2: 控制风格信息的影响
        :return:
        '''
        ## 跳跃连接
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w1 * (dec_feat * scale + shift)

        ## image light prompt
        x1 = dec_feat + residual
        prompt_light = self.prompt_light(x1, test=test)
        fuse1 = torch.cat([prompt_light, x1], 1)
        residual2 = self.fuse(self.reduce_channel(fuse1))

        # 用于可视化
        self.pre_prompt = x1
        self.post_prompt = prompt_light
        self.residual2 = residual2

        ## image prompt
        # prompt_input = self.encode_enc2(torch.cat([style_feat, dec_feat], dim=1))
        # prompt = self.prompt(style_feat)
        # fuse2 = torch.cat([style_feat, prompt], 1)
        # residual3 = self.fuse2(self.reduce_channel2(fuse2))

        out = dec_feat + residual + residual2

        return out


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self,
                 n_e,           # codebook 的长度
                 e_dim,         # codebook 每个特征的维度
                 weight_path=r'/home/wuxu/codes/RIDCP/pretrain_networks/weight_for_matching_dehazing_Flickr.pth',
                 beta=0.25,
                 LQ_stage=False,
                 use_weight=True,
                 weight_alpha=1.0):
        super().__init__()
        self.codebook_size = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.use_weight = use_weight
        self.weight_alpha = weight_alpha
        if self.use_weight:
            self.weight = nn.Parameter(torch.load(weight_path))
            self.weight.requires_grad = False
        self.embedding = nn.Embedding(self.codebook_size, self.e_dim)

        self.register_buffer('index_counts', torch.zeros(1024, dtype=torch.int))


    def dist(self, x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None, weight_alpha=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # -------------------------------------------------------
        # 拉平 z，获得codebook
        # reshape z -> (batch, height, width, channel) and flatten
        # -------------------------------------------------------
        # print(f'z: {z.shape}')
        z = z.permute(0, 2, 3, 1).contiguous()
        # print(f'z after: {z.shape}')
        # print('000000000000000')
        # print(self.e_dim)
        z_flattened = z.view(-1, self.e_dim)
        # print(f'z_flattened: {z_flattened.shape}')

        codebook = self.embedding.weight
        # print(z_flattened.size())
        # print(f'codebook: {codebook.shape}')

        # -------------------------------------------------------
        # 计算 特征与codebook中量化特征之间的距离
        # -------------------------------------------------------
        d = self.dist(z_flattened, codebook)    # d: [256960, 1024]
        
        # -------------------------------------------------------
        # CHM : Controllable HQPs Matching
        # 仅在测试时使用
        # -------------------------------------------------------
        # if self.use_weight and self.LQ_stage:
        #     if weight_alpha is not None:
        #         self.weight_alpha = weight_alpha
        #     d = d * torch.exp(self.weight_alpha * self.weight)

        # -------------------------------------------------------
        # 根据计算的距离，获得z对应量化特征的下标 index
        # find closest encodings
        # -------------------------------------------------------
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)   #
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)


        # print(f'd shape: {d.shape}')
        # print(f'min_encoding_indices shape: {min_encoding_indices.shape}')
        # print(f'min_encodings shape: {min_encodings.shape}')
        # print(f'min_encoding_indices: {min_encoding_indices}')
        # print(f'min_encodings: {min_encodings}')

        # 计算范围0到1024内每个索引的出现次数
        self.index_counts = torch.bincount(min_encoding_indices.squeeze(), minlength=1024)
        # self.total_index_counts += index_counts

        # -------------------------------------------------------
        # 获得 GT 的 量化特征，用来计算 codebook loss
        # -------------------------------------------------------
        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # -------------------------------------------------------
        # 获得输入图像的量化特征，get quantized latent vectors
        # -------------------------------------------------------
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        # -------------------------------------------------------
        # 训练 Stage I使用到的 特征损失
        # -------------------------------------------------------
        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            # 训练 Stage II
            # codebook_loss = self.dist(z_q, z_q_gt.detach()).mean() \
                            # + self.beta * self.dist(z_q_gt.detach(), z)
            codebook_loss = self.beta * self.dist(z_q_gt.detach(), z)
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            # print("codebook loss:", codebook_loss.mean(), "\ntexture_loss: ", texture_loss.mean())
            codebook_loss = codebook_loss + texture_loss
        else:
            # 训练 Stage I
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256,
                blk_depth=6,
                num_heads=8,
                window_size=8,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 prompt_dim=64,
                 prompt_len=5,
                 prompt_size = 64,
                 lin_dim = 96,
                 num_heads=1, 
                 ffn_expansion_factor=2.66, 
                 num_blocks=1, 
                 LayerNorm_type='WithBias', 
                 bias=False,
                 **swin_opts,
                 ):
        super().__init__()
        self.LQ_stage = LQ_stage
        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res

        self.prompt_blocks = nn.ModuleList()
        self.transf_blocks = nn.ModuleList()
        self.reduce_channel_blocks = nn.ModuleList()

        for i in range(max_depth):

            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            res = res // 2
            tmp_down = nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1)
            # tmp_down = nn.MaxPool2d(2,2)
            self.down_blocks.append(tmp_down)

            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            
            self.prompt_blocks.append(PromptGenBlock(prompt_dim=prompt_dim[res],
                prompt_len=prompt_len[res],
                prompt_size = prompt_size[res],
                lin_dim = lin_dim[res]))

            tb = [TransformerBlock(dim=out_ch, 
                num_heads=num_heads[res], 
                ffn_expansion_factor=ffn_expansion_factor, 
                bias=bias, 
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[res])]
            self.transf_blocks.append(nn.Sequential(*tb))
            self.reduce_channel_blocks.append(nn.Conv2d(out_ch * 2, out_ch,kernel_size=1,bias=bias))
        
        # if LQ_stage:
        #     self.blocks.append(SwinLayers(**swin_opts))

    def forward(self, input):
        # 原始的 VQGAN 结构和调用
        # 如果要在 encoder中添加分支新结构，需要重新写个方法。不要改变这个，因为 获得 GT indice的时候，用的是这个方法
        # input.requires_grad = True
        x = self.in_conv(input)

        for idx, m in enumerate(self.blocks):
            with torch.backends.cudnn.flags(enabled=False):
                # 并行结构
                x = m(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


@ARCH_REGISTRY.register()
class LightQANet_Arch(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=[[64, 1024, 512]],
                 gt_resolution=256,
                 LQ_stage=True,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_semantic_loss=False,
                 use_Latent_ContrastLoss=False,
                 use_weight=False,
                 weight_alpha=1.0,
                 seg_cfg=None,
                 weight_texture=1,
                 weight_style=1,
                 weight_light=1,
                 prompt_len=5,
                 **ignore_kwargs):

        super().__init__()

        # print(f'weight_light: {weight_light}')
        # sys.exit()

        codebook_params = np.array(codebook_params)
        self.prompt_len = prompt_len

        self.codebook_scale = codebook_params[:, 0]             # 64
        codebook_emb_num = codebook_params[:, 1].astype(int)    # 1024
        codebook_emb_dim = codebook_params[:, 2].astype(int)    # 512

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        # self.use_residual = use_residual
        # self.only_residual = only_residual
        self.use_weight = use_weight
        # self.use_warp = use_warp
        self.weight_alpha = weight_alpha
        self.weight_texture = weight_texture
        self.weight_style = weight_style
        self.weight_light = weight_light
        self.use_Latent_ContrastLoss = use_Latent_ContrastLoss

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        prompt_dim_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        prompt_len_dict = {
            8: prompt_len,
            16: prompt_len,
            32: prompt_len,
            64: prompt_len,
            128: prompt_len,
            256: prompt_len,
            512: prompt_len,
        }

        prompt_size_dict = {
            8: 8,
            16: 8,
            32: 8,
            64: 16,
            128: 32,
            256: 64,
            512: 64,
        }

        # prompt中全连接层的输入维度，维度与 prompt 输入特征的保持一致
        lin_dim_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # transformer
        num_head_dict = {
            8: 8,
            16: 8,
            32: 8,
            64: 4,
            128: 2,
            256: 1,
            512: 1,
        }

        # num_blocks_dict = {
        #     8: 6,
        #     16: 6,
        #     32: 8,
        #     64: 6,
        #     128: 6,
        #     256: 4,
        #     512: 4,
        # }

        num_blocks_dict = {
            8: 4,
            16: 4,
            32: 4,
            64: 3,
            128: 3,
            256: 2,
            512: 2,
        }

        region_size_dict = {
            32: 2,
            64: 2,
            128: 4,
            256: 4,
        }


        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        self.multiscale_encoder = MultiScaleEncoder(
                                in_channel,
                                self.max_depth,
                                self.gt_res,
                                channel_query_dict,
                                norm_type, act_type, LQ_stage,
                                prompt_dim=prompt_dim_dict,
                                prompt_len=prompt_len_dict,
                                prompt_size = prompt_size_dict,
                                lin_dim = lin_dim_dict,
                                num_heads= num_head_dict, 
                                ffn_expansion_factor=2.66, 
                                num_blocks=num_blocks_dict, 
                                LayerNorm_type='WithBias', 
                                bias=False,
                                )

        # self.context_module = SwinLayers()

        self.use_semantic_loss = use_semantic_loss

        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.decoder_group_frozen = copy.deepcopy(self.decoder_group)
        for param in self.decoder_group_frozen.parameters():
             param.requires_grad = False
        
        # 打印原始模型和复制模型的参数，验证它们是否相同
        # for param_orig, param_copy in zip(self.decoder_group.parameters(), self.decoder_group_copy.parameters()):
        #     print(torch.equal(param_orig.data, param_copy.data))
        
        # sys.exit()

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        # self.residual_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # fuse_conv_dict
        self.connect_list = [64, 128]
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = channel_query_dict[f_size]
            self.fuse_convs_dict[str(f_size)] = Fuse_aft_block(in_ch, in_ch, 
                embed_dim=lin_dim_dict[f_size],
                num_heads=num_head_dict[f_size],
                num_blocks=num_blocks_dict[f_size],
                region_size=region_size_dict[f_size],
                prompt_dim=prompt_dim_dict[f_size],
                prompt_len=prompt_len_dict[f_size],
                prompt_size=prompt_size_dict[f_size],
                lin_dim=lin_dim_dict[f_size],
                )

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        # 对特征进行量化操作，codebook_params.shape[0] = 1
        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
                LQ_stage=self.LQ_stage,
                use_weight=self.use_weight,
                weight_alpha=self.weight_alpha
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim[scale]
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]

            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))

        # self.total_index_counts = self.quantize_group[0].index_counts
        self.visual_feature_num = 0
        self.idx = 0
        # 初始化 CLIP
        # self._init_CLIP()

        ## 统计参数量
        # p_num1 = self.count_learnable_parameters(self.multiscale_encoder)
        # p_num2 = self.count_learnable_parameters(self.fuse_convs_dict['64'])
        # p_num3 = self.count_learnable_parameters(self.fuse_convs_dict['128'])

        # total_params = p_num1 + p_num2 + p_num3
        # if total_params < 1e3:
        #     total_params =  f"{total_params} params"
        # elif total_params < 1e6:
        #     total_params =  f"{total_params / 1e3:.2f}K params"
        # elif total_params < 1e9:
        #     total_params = f"{total_params / 1e6:.2f}M params"
        # else:
        #     total_params = f"{total_params / 1e9:.2f}G params"
        
        # print(f'learnable parameters: {total_params}')

        # import sys
        # sys.exit()

    def count_learnable_parameters(self, module):
        """
        计算并返回给定模块中可学习参数的总数，并以易读的格式输出。
        """
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return total_params

    def _init_CLIP(self):
        if clip is None:
            raise ImportError('clip is required when CLIP features are enabled.')
        # CLIP
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

    def _CLIP_encoder(self, x, is_img=True):  
        if clip is None:
            raise ImportError('clip is required when CLIP features are enabled.')

        # CLIP 提取文本、图像信息

        if is_img:
            # image = self.clip_preprocess(x)
            image224 = F.interpolate(x, size=(224, 224))        # CLIP 的图像输入尺寸为 224*224
            feature = self.clip_model.encode_image(image224)
        else:
            text = clip.tokenize(x).cuda()
            feature = self.clip_model.encode_text(text)

        return feature

    def _visual_feature(self, input):
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        from scipy.stats import norm
        import matplotlib.pyplot as plt

        self.visual_feature_num = self.visual_feature_num + 1

        # for i in range(5):
        for j in range(10):
            plt.figure(figsize=(10, 10))
            feature_maps = input
            #print(feature_maps.size())
            selected_feature_map = feature_maps[0,j,:,:].cpu().detach().numpy()
            norm = Normalize(vmin=selected_feature_map.min(), vmax=selected_feature_map.max())

            # 使用颜色映射 'viridis' 并且根据数据的值来映射颜色
            plt.imshow(selected_feature_map, cmap='bwr', norm=norm)
            plt.axis('off')  # 关闭坐标轴
            plt.tight_layout()      # 自动调整布局，减少空白边缘
            #plt.title(f'Batch {i}, Channel {j}')
                
            # 保存为不同的文件，可以根据需要调整文件名
            file_name = f'/data/data0/wuxu/codes/low_light/PromptEnhance/visual_net/visual_feature/decoder_last_residual2_LLIE_Prior_PromptV2_5_Arch_LOLv2Real_All_20240811_105650/fig_{self.visual_feature_num}_ch_{j}_LLIE_Prior_PromptV2_5_Arch_LOLv2Real_All_20240811_105650.png'
            plt.savefig(file_name)
            plt.close()
            

        print("Feature maps saved.")
        if self.visual_feature_num == 10:
            sys.exit()

    def _visual_prompt(self, input, name=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        self.visual_feature_num += 1
        b, c, h, w = input.shape
        num_prompts, channels = b, c
        in_np = input.cpu().detach().numpy()
        # 将每个 prompt 中的每个通道展平为一个 1024 维向量，总共 5*256 = 1280 个数据点
        data_flat = in_np.reshape(num_prompts * channels, -1)

        # 构造标签，每个通道对应其所属的 prompt（标签 0~4）
        labels = np.repeat(np.arange(num_prompts), channels)

        # 使用 t-SNE 将 1024 维数据降到 2 维，设置 perplexity 为 30
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)  
        data_tsne = tsne.fit_transform(data_flat)

        # 绘制 t-SNE 的散点图，不同颜色代表不同的 prompt
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', s=10)
        plt.colorbar(sc, ticks=range(num_prompts))
        plt.title("t-SNE of prompts")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        if name is not None:
            plt.savefig(f'/data/data0/wuxu/codes/low_light/PromptEnhance/visual_net/visual_prompt/t_sne_prompt_{name}.png')
        else:
            plt.savefig(f'/data/data0/wuxu/codes/low_light/PromptEnhance/visual_net/visual_prompt/t_sne_prompt{self.visual_feature_num}.png')
        # plt.show()

        ## 所有prompt的通道之间的相似度
        corr_matrix = np.corrcoef(data_flat)

        # 在第 i 个子图上用 imshow 绘制相关系数矩阵
        plt.figure(figsize=(8, 6))
        # 设置 vmin=-1, vmax=1 以便固定相关系数在 [-1, 1] 范围
        im = plt.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1, aspect='equal')
        plt.title(f"All Prompts Channel Similarity")
        plt.xlabel("Channel")
        plt.ylabel("Channel")

        plt.colorbar(im)
        plt.savefig(f'/data/data0/wuxu/codes/low_light/PromptEnhance/visual_net/visual_prompt/correct_all_prompt_{name}.png')
        # 为每个子图添加一个 colorbar
        # cbar = fig.colorbar(im, ax=ax)
        # cbar.set_label("Correlation")


        # ========== 2. 绘制通道间相似度矩阵，使用 Matplotlib imshow ==========
        fig, axes = plt.subplots(1, num_prompts, figsize=(4 * num_prompts, 4))

        for i in range(num_prompts):
            # 取第 i 个 Prompt 的所有通道特征，形状 (channels, height, width)
            prompt_i = in_np[i]
            # 拉平为 (channels, height*width)
            prompt_i_flat = prompt_i.reshape(channels, -1)

            # 计算通道之间的 Pearson 相关系数矩阵，形状 (channels, channels)
            corr_matrix = np.corrcoef(prompt_i_flat)

            # 在第 i 个子图上用 imshow 绘制相关系数矩阵
            ax = axes[i]
            # 设置 vmin=-1, vmax=1 以便固定相关系数在 [-1, 1] 范围
            im = ax.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1, aspect='equal')
            ax.set_title(f"Prompt {i} Channel Similarity")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Channel")
            
            # 为每个子图添加一个 colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Correlation")

        plt.tight_layout()
        plt.savefig(f'/data/data0/wuxu/codes/low_light/PromptEnhance/visual_net/visual_prompt/correct_prompt_{name}.png')
        # plt.show()

        # ========== 3. 分开绘制每个 prompt 的 2D t-SNE，可观察 prompt 内部差异 ==========
        # 我们可以对每个 prompt 的 256 个通道单独进行 2D t-SNE
        # 也可以统一做一次 2D t-SNE，再根据 prompt 分组绘图
        # 这里示例逐个 prompt 做单独降维，以更好观察各自分布

        fig, axes = plt.subplots(1, num_prompts, figsize=(4*num_prompts, 4), sharex=False, sharey=False)

        for i in range(num_prompts):
            # 取第 i 个 prompt 的数据，形状 (256, 32, 32)
            prompt_i = in_np[i]
            # 展平为 (256, 1024)
            prompt_i_flat = prompt_i.reshape(channels, -1)
            # 进行 2D t-SNE
            tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
            prompt_i_tsne_2d = tsne_2d.fit_transform(prompt_i_flat)
            
            # 绘制散点图
            ax = axes[i]
            sc_i = ax.scatter(prompt_i_tsne_2d[:, 0], prompt_i_tsne_2d[:, 1],
                            c=np.arange(channels), cmap='viridis', s=10)
            ax.set_title(f"Prompt {i} - 2D t-SNE")
            ax.set_xlabel("TSNE-1")
            ax.set_ylabel("TSNE-2")
            plt.colorbar(sc_i, ax=ax, label="Channel Index")

        plt.tight_layout()
        plt.savefig(f'/data/data0/wuxu/codes/low_light/PromptEnhance/visual_net/visual_prompt/t_sne_per_prompt_{name}.png')
        # plt.show()


    def _embed_semantic_before_context(self, encode_feat, semantic_feat):
        '''
        将语义信息与encoder的输入融合
        :param x:
        :param y:
        :return:
        '''
        feat_size = encode_feat.size()[2:]

        semantic_feat = F.interpolate(semantic_feat, size=feat_size)
        semantic_feat = self.resize_semantic_feat_conv(semantic_feat)

        fuse_info = self.fuse_semantic_encode(torch.cat((encode_feat,semantic_feat), 1))

        return fuse_info


    def _encode_reference_feats(self, input, net_hq):

        enc_feat_dict = {}

        x= net_hq.multiscale_encoder.in_conv(input)
        for idx, m in enumerate(net_hq.multiscale_encoder.blocks):
            cur_res = self.gt_res // 2 ** self.max_depth * 2 ** (1-idx)
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)
                enc_feat_dict[str(cur_res)] = x.clone()
        # print(enc_feat_dict)

        return enc_feat_dict


    def _get_semantic_info(self, x):
        '''
        利用现有的网络，获得图像的语义信息
        :param x:
        :return:
        '''

        out = {}
        out['vgg_feats'] = None
        out['seg_feats'] = None

        # vgg
        if x is not None:
            with torch.no_grad():
                vgg_feats = self.vgg_feat_extractor(x)[self.vgg_feat_layer]
                out['vgg_feats'] = vgg_feats
                # mobilnet
                #   deeplabv3plus: seg_feat = {'high_level_features': 'out', 'low_level_features': 'low_level'}
                #   deeplabv3: seg_feat = {'high_level_features': 'out'}
                # resnet
                #   deeplabv3plus: seg_feat = {'layer4': 'out', 'layer1': 'low_level'}
                #   deeplabv3: seg_feat = {'layer4': 'out'}
                seg_feats = self.seg_feat_extractor(x)
                out['seg_feats'] = seg_feats['out']

        return out


    def encode_indices(self, input, weight_alpha=None):

        '''
        用于获得 input 的 量化特征 的 indices，这部分只能用到 VQGAN stage1 中训练到的结构！
        :param input:
        :return:
        '''

        indices_list = []

        ## 特征提取
        enc_feat_dict = {}
        x = self.multiscale_encoder.in_conv(input)
        for idx, m in enumerate(self.multiscale_encoder.blocks):
            cur_res = self.gt_res // 2 ** self.max_depth * 2 ** (1 - idx)
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)
                enc_feat_dict[str(cur_res)] = x.clone()

        enc_feats = x
        enc_feat_dict['enc_feats'] = enc_feats

        # enc_feats = self.multiscale_encoder(input)

        ## 量化特征
        feat_to_quant = self.before_quant_group[0](enc_feats)

        # 量化特征：获得量化特征、计算 codebook loss
        if weight_alpha is not None:
            self.weight_alpha = weight_alpha
        z_quant, codebook_loss, indices = self.quantize_group[0](feat_to_quant, weight_alpha=self.weight_alpha)

        after_quant_feat = self.after_quant_group[0](z_quant, None)

        indices_list.append(indices)

        return indices_list, z_quant, feat_to_quant, after_quant_feat, enc_feat_dict


    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)

        return out_img

    
    def encoder(self, input):
        '''
        encoder部分，便于调用。但是由于之前的代码都是写到一起的，为了方便后面模型的测试，对 encode_and_decode() 不做改动
        :param input:
        :param reference_img:
        :param weight_alpha:
        :param net_hq:
        :return:
        '''
        # --------------------------------------------
        # encoder：获得图像特征
        # --------------------------------------------
        enc_feat_dict = {}
        x = self.multiscale_encoder.in_conv(input)
        for idx, m in enumerate(self.multiscale_encoder.blocks):
            cur_res = self.gt_res // 2 ** self.max_depth * 2 ** (1 - idx)
            with torch.backends.cudnn.flags(enabled=False):

                x1 = m(x)

                x2 = self.multiscale_encoder.down_blocks[idx](x)
                x2 = self.multiscale_encoder.prompt_blocks[idx](x2)
                fuse = torch.cat([x1, x2], 1)
                x = self.multiscale_encoder.reduce_channel_blocks[idx](fuse)
                x = self.multiscale_encoder.transf_blocks[idx](x)

                enc_feat_dict[str(cur_res)] = x.clone()


        enc_feats = x

        enc_feat_dict['enc_feats'] = enc_feats

        return enc_feat_dict


    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height
        output_width = width
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile, _ = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x
                output_end_x = input_end_x
                output_start_y = input_start_y
                output_end_y = input_end_y

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad)
                output_end_x_tile = output_start_x_tile + input_tile_width
                output_start_y_tile = (input_start_y - input_start_y_pad)
                output_end_y_tile = output_start_y_tile + input_tile_height

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                       output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                  output_start_x_tile:output_end_x_tile]
        return output


    @torch.no_grad()
    def test(self, input, reference, net_hq, weight_alpha=None, idx=0):

        self.idx = idx  # 第几张图片
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False

        # padding to multiple of window_size * 8
        wsz = 32
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        if reference is not None:
            reference = torch.cat([reference, torch.flip(reference, [2])], 2)[:, :, :h_old + h_pad, :]
            reference = torch.cat([reference, torch.flip(reference, [3])], 3)[:, :, :, :w_old + w_pad]

        outdict= self.encode_and_decode(input=input,
                                        reference_img=reference,
                                        net_hq=net_hq,
                                        weight_alpha=weight_alpha)
        
        self.total_index_counts = self.quantize_group[0].index_counts
        output = outdict['out_img']
        if output is not None:
            output = output[..., :h_old, :w_old]
        self.use_semantic_loss = org_use_semantic_loss

        return output


    def encode_and_decode(self, input, gt_img=None, reference_img=None, gt_indices=None, weight_alpha=None, net_hq=None, prompt_text=None):

        # --------------------------------------------
        # encoder：获得图像特征
        # --------------------------------------------

        enc_feat_dict = self.encoder(input)
        enc_feats = enc_feat_dict['enc_feats']

        # --------------------------------------------
        # image prompt 分支：获得图像提示信息
        # --------------------------------------------
        if reference_img is not None and net_hq is not None:
            if reference_img.shape[:2] != input.shape[:2]:
                reference_img = F.interpolate(reference_img, input.shape[:2])
            enc_feats_refer_dict = self._encode_reference_feats(reference_img, net_hq)
        else:
            enc_feats_refer_dict = enc_feat_dict

        codebook_loss_list = []
        indices_list = []
        semantic_loss_list = []
        code_decoder_output = []

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        out_img = None

        de_prompt = []
        en_prompt = []

        # --------------------------------------------
        # Decoder：VQ-GAN  stage I 训练好的 decoder 重构图像
        # 输入：为encoder的特征，以及经过量化的特征。Note：将多个level的特征量化后送入相应Decoder层
        # 输出：重构图像
        # --------------------------------------------
        x = enc_feats
        # print(f'enc_feats: {enc_feats_context.shape}')
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i

            # 如果此时输入特征的长度在 [64, 1024, 512] 中，则将该特征量化
            if cur_res in self.codebook_scale:  # needs to perform quantize

                # 获得用于量化的特征：将输入特征
                if prev_dec_feat is not None:
                    # 将decoer上一个level的输入特征与当前输入特征做拼接，相当于一个跳跃连接了。
                    before_quant_feat = torch.cat((x, prev_dec_feat), dim=1)
                else:
                    before_quant_feat = x
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)

                # 量化特征：获得量化特征、计算codebook loss
                if weight_alpha is not None:
                    self.weight_alpha = weight_alpha
                if gt_indices is not None:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, gt_indices[quant_idx], weight_alpha=self.weight_alpha)
                else:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, weight_alpha=self.weight_alpha)

                
                # 语义损失，在中间特征层中计算
                if self.use_semantic_loss:
                    semantic_z_quant = self.conv_semantic(z_quant)
                    semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
                    semantic_loss_list.append(semantic_loss)

                if not self.use_quantize:
                    z_quant = feat_to_quant

                after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)

                codebook_loss_list.append(codebook_loss)
                indices_list.append(indices)

                quant_idx += 1
                prev_quant_feat = z_quant
                x = after_quant_feat

            # 跳跃连接
            x_frozen = self.decoder_group_frozen[i](x)
            test=False
            if i == 1:
                test=True
            x = self.fuse_convs_dict[str(cur_res)](enc_feat=enc_feat_dict[str(cur_res)],
                                                #    enc_feat=enc_feat_dict[str(cur_res)].detach(),
                                                #    style_feat=enc_feats_refer_dict[str(cur_res)].detach(),
                                                   style_feat=x_frozen.detach(),
                                                   dec_feat=x,
                                                   w1=self.weight_texture,
                                                   test=test)
            # 绘制 prompt
            # x2 = self.fuse_convs_dict[str(cur_res)].prompt_light.prompt_param
            # p = x2.squeeze(0)
            # name = f'de_{cur_res}'
            # self._visual_prompt(p, name)

            # p = self.fuse_convs_dict[str(cur_res)].post_prompt
            # p = p.cpu().detach().numpy()
            # sp = f'visual_net/visial_prompt/dep{i}'
            # if not os.path.exists(sp):
            #     os.makedirs(sp)
            # np.save(f'{sp}/{self.idx}_de_prompt_{i}.npy', p)

            ## prompt weights
            # p = self.fuse_convs_dict[str(cur_res)].prompt_light.prompt_weights
            # p = p.cpu().detach().numpy()
            # sp = f'visual_net/prompt_weights/dep{i}'
            # if not os.path.exists(sp):
            #     os.makedirs(sp)
            
            # 存储到 .npy 文件
            # np.save(f'{sp}/{self.idx}_de_prompt_weights_{i}.npy', p)

            x = self.decoder_group[i](x)
            code_decoder_output.append(x)
            prev_dec_feat = x
            
        # 可视化特征图，临时代码，做图用
        # self._visual_feature(self.fuse_convs_dict[str(cur_res)].residual2)

        out_img = self.out_conv(x)


        ## prompt 分析的
        # for idx, m in enumerate(self.multiscale_encoder.blocks):
        #     p = self.multiscale_encoder.prompt_blocks[idx].prompt
        #     p = p.cpu().detach().numpy()

        #     sp = f'visual_net/visial_prompt/enp{idx}'
        #     if not os.path.exists(sp):
        #         os.makedirs(sp)

        #     np.save(f'{sp}/{self.idx}_en_prompt_{idx}.npy', p)
            
            ## prompt weights
            # p = self.multiscale_encoder.prompt_blocks[idx].prompt_weights
            # p = p.cpu().detach().numpy()
            # sp = f'visual_net/prompt_weights/enp{idx}'
            # if not os.path.exists(sp):
            #     os.makedirs(sp)
            
            # 存储到 .npy 文件
            # np.save(f'{sp}/{self.idx}_en_prompt_weights_{idx}.npy', p)

        

        if len(codebook_loss_list) > 0:
            codebook_loss = sum(codebook_loss_list)
        else:
            codebook_loss = 0
        semantic_loss = sum(semantic_loss_list) if len(semantic_loss_list) else codebook_loss * 0

        out_dict = {}
        out_dict['out_img'] = out_img
        out_dict['codebook_loss'] = codebook_loss
        out_dict['enc_feat_dict'] = enc_feat_dict
        out_dict['semantic_loss'] = semantic_loss
        out_dict['feat_to_quant'] = feat_to_quant
        out_dict['after_quant_feat'] = after_quant_feat
        out_dict['z_quant'] = z_quant
        out_dict['indices_list'] = indices_list

        return out_dict


    def forward(self, input, gt_img=None, reference_img=None, gt_indices=None, weight_alpha=None, net_hq=None, prompt_text=None):

        # --------------------------------------------------------------
        # 训练 stage II，获得gt的量化特征下标，用于特征量化的监督
        # in LQ training stage, need to pass GT indices for supervise.
        # --------------------------------------------------------------
        # if gt_indices is not None:
        outdict = self.encode_and_decode(input=input,
            gt_img=gt_img,
            reference_img=reference_img,
            gt_indices=gt_indices,
            weight_alpha=weight_alpha,
            net_hq=net_hq,
            prompt_text=prompt_text)

        # --------------------------------------------------------------
        # 测试阶段
        # in HQ stage, or LQ test stage, no GT indices needed.
        # --------------------------------------------------------------
        # else:
        #     dec, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices = self.encode_and_decode(input, reference_img, weight_alpha=weight_alpha,net_hq=net_hq)

        return outdict
