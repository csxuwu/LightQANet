import glob
import os.path
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as tvu
from torch.autograd import Variable
import copy

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.metrics import calculate_metric
from basicsr.data.random_load_images import random_load_images
from basicsr.archs.filters_lowlight import Low_Level_Filter_conv, Low_Level_Filter2_conv
from basicsr.archs.LatentEnhance.LLF_arch import LLF_Arch

# from basicsr.losses import fifo_losses
from basicsr.pytorch_metric_learning import losses
from basicsr.pytorch_metric_learning.distances import CosineSimilarity
from basicsr.pytorch_metric_learning.reducers import MeanReducer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import cv2



# -----------------------------------------
# 2024-05-16
# 用来训练 light pass filter
# feature extractor : codebook encoder

# RGB、bright、contrast 的计算不同，去除零像素
# -----------------------------------------

def get_avg_rgb(image, mask_512=None):
    '''
    计算RGB个通道的均值
    '''

    if mask_512 is not None:
        image = image * mask_512

    average_rgb = image.mean(dim=-1).mean(dim=-1)

    return average_rgb

def get_average_brightness(image):
    '''
    计算平均亮度
    '''
    # 去除纯黑部分，这部分通常不属于亮度范围，而是物体本身的颜色。防止背景噪声影响整体的亮度计算，从而得到更准确的对比度和亮度信
    non_zero_pixels = image[image != 0]
    average_brightness = torch.mean(non_zero_pixels.float())
    return average_brightness

def get_contrast(image):
    '''
    计算对比度
    '''
    # Find non-zero pixels
    non_zero_pixels = image[image != 0].float()
    if non_zero_pixels.numel() == 0:
        return torch.tensor(0.0)
    # Calculate the mean intensity of the non-zero pixels
    mean_intensity = torch.mean(non_zero_pixels)
    # Calculate the squared difference from the mean for each pixel
    squared_diff = (non_zero_pixels - mean_intensity) ** 2
    # Calculate the variance (average of squared differences)
    variance = torch.mean(squared_diff)
    # Contrast is the square root of variance
    contrast = torch.sqrt(variance)
#     print("contrast",contrast)
    return contrast  

def gram_matrix(input):
    """
    计算给定特征的Gram矩阵。
    参数:
        input: 四维的张量，形状为 (N, C, H, W)
    返回:
        Gram矩阵，形状为 (N, C, C)
    """
    # 获取各维度
    a, b, c, d = input.size()
    
    # 改变形状：将 (N, C, H, W) 转换为 (N, C, H*W)
    features = input.view(a, b, c * d)
    
    # 计算Gram矩阵：使用批次矩阵乘法 bmm
    G = torch.bmm(features, features.transpose(1, 2))  # 交换C和H*W的维度
    
    # 标准化Gram矩阵的值，除以每个特征图的元素数量
    G = G / (c * d)
    
    return G


def weightedMSE(D_out, D_label):
    return torch.mean((D_out - D_label).abs() ** 2)


def wasserstein1d(x, y, aggregate=True):
    """Compute wasserstein loss in 1D"""
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    n = x.size(0)
    if aggregate:
        z = (x1-y1).view(-1)
        return torch.dot(z, z)/n
    else:
        return (x1-y1).square().sum(0)/n


def quantization_swdc_loss(b, device='cuda', aggregate=True):
    real_b = torch.randn(b.shape, device=device).sign()
    bsize, dim = b.size()

    if aggregate:
        gloss = wasserstein1d(real_b, b) / dim
    else:
        gloss = wasserstein1d(real_b, b, aggregate=False)

    return gloss



@MODEL_REGISTRY.register()
class Low_Level_Filter2_Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.feature_extractor = build_network(opt['network_g'])
        self.feature_extractor = self.model_to_device(self.feature_extractor)

        # 几个 low level 信息提取器
        lr_fpf1 = 1e-3 
        self.LLF_filters = LLF_Arch(256, 1) 
        self.LLF_filters_optimizer = torch.optim.Adamax([p for p in self.LLF_filters.parameters() if p.requires_grad == True], lr=lr_fpf1)
        self.LLF_filters.to(self.device)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_hq', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.feature_extractor, load_path, self.opt['path']['strict_load'])

        load_path = self.opt['path'].get('pretrain_network_LLF', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading LLF from {load_path}')
            self.load_network(self.LLF_filters, load_path, self.opt['path']['strict_load'])

        self.setup_schedulers()
        self.l2 = nn.MSELoss()
        

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.b,_,_,_ = self.lq.shape
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = None

        if 'refer' in data:
            self.refer = data['refer'].to(self.device)
        else:
            self.refer = None

        if 'gt_caption' in data:
            self.gt_caption = data['gt_caption']
            # print(self.gt_caption)
        else:
            self.gt_caption = None

        if 'mean_r' in data:
            self.mean_r = data['mean_r'].to(self.device)
            self.mean_g = data['mean_g'].to(self.device)
            self.mean_b = data['mean_b'].to(self.device)
            self.light = data['light'].to(self.device)
            self.global_contrast = data['global_contrast'].to(self.device)

    def convert_value_to_map(self, a, value_list):
        N, C, H, W = a.shape
        value_list = value_list.cpu().numpy()
        t = [torch.full((1, C, H, W), value, requires_grad=True) for value in value_list]
        t2 = torch.cat(t, dim=0)
        t2 = t2.to(self.device)

        return t2

    def optimize_parameters(self, current_iter):

        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.LLF_filters.parameters():
            p.requires_grad = True

        # 梯度清零
        self.LLF_filters_optimizer.zero_grad()

        loss_dict = OrderedDict()

        _,_,_,_, lq_feat_dict = self.feature_extractor.encode_indices(self.lq)

        lq_feat0 = lq_feat_dict['128'].detach()
        lq_feat1 = lq_feat_dict['64'].detach()
        self.lq_feats = {'layer0': lq_feat0, 'layer1': lq_feat1}

        total_lf_loss = 0.0

        # lq_feat0_gram = gram_matrix(lq_feat0)
        # lq_feat0_gram = lq_feat0_gram.unsqueeze(1)
        
        ## 计算 low level 参数
        color_value, contrast_value, light_value = self.LLF_filters(lq_feat1)
        
        avg_rgb_gt = get_avg_rgb(self.lq)
        contrast_gt = get_contrast(self.lq)
        light_gt = get_average_brightness(self.lq)

        rgb_loss = self.l2(color_value, avg_rgb_gt)
        contrast_loss = self.l2(contrast_value, contrast_gt)
        light_loss = self.l2(light_value, light_gt)

        total_lf_loss = rgb_loss + contrast_loss + light_loss

        # 计算梯度
        total_lf_loss.backward(retain_graph=False)
        loss_dict['total_lf_loss'] = total_lf_loss

        # 更新模型参数
        self.LLF_filters_optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.light_pass_filter_loss_dict = loss_dict


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def test_single_img(self, lq_img_path, refer_img_path):

        lq_img = cv2.imread(lq_img_path)
        refer_img = cv2.imread(refer_img_path)

        lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
        lq_img = torch.from_numpy(lq_img)
        lq_img = lq_img.permute(2, 0, 1)
        lq_img = lq_img.unsqueeze(0)
        lq_img = lq_img.to(self.device)

        refer_img = cv2.cvtColor(refer_img, cv2.COLOR_BGR2RGB)
        refer_img = torch.from_numpy(refer_img)
        refer_img = refer_img.permute(2, 0, 1)
        refer_img = refer_img.unsqueeze(0)
        refer_img = refer_img.to(self.device)

        self.feature_extractor.eval()
        self.R_filter.eval()
        self.G_filter.eval()
        self.B_filter.eval()
        self.Light_filter.eval()
        self.Contrast_Global_filter.eval()

        _,_,_,_, refer_feat_dict = self.feature_extractor.encode_indices(refer_img)

        ref_feat0 = refer_feat_dict['128'].detach()
        ref_feat1 = refer_feat_dict['64'].detach()
        self.lq_feats = {'layer0': ref_feat0, 'layer1': ref_feat1}

        ref_feat0_gram = gram_matrix(ref_feat0)
        ref_feat0_gram = ref_feat0_gram.unsqueeze(1)

        # r_value = self.R_filter(ref_feat0_gram)
        # g_value = self.G_filter(ref_feat0_gram)
        # b_value = self.B_filter(ref_feat0_gram)
        # light_value = self.Light_filter(ref_feat0_gram)
        # cg_value = self.Contrast_Global_filter(ref_feat0_gram)


    def save(self, epoch, current_iter):

        self.save_network(self.LLF_filters, 'LLF_filters', current_iter)
        # self.save_network(self.G_filter, 'G_filter', current_iter)
        # self.save_network(self.B_filter, 'B_filter', current_iter)
        # self.save_network(self.Light_filter, 'Light_filter', current_iter)
        # self.save_network(self.Contrast_Global_filter, 'Contrast_Global_filter', current_iter)


        self.save_training_state(epoch, current_iter)



