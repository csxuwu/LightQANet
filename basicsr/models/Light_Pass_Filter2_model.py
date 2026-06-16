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
from basicsr.archs.discriminator_arch import LightPassFilter_conv1, LightPassFilter_res1
# from basicsr.losses import fifo_losses
from basicsr.pytorch_metric_learning import losses
from basicsr.pytorch_metric_learning.distances import CosineSimilarity
from basicsr.pytorch_metric_learning.reducers import MeanReducer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



# -----------------------------------------
# 2024-05-16
# 用来训练 light pass filter
# feature extractor: backbone_yolo v7
# -----------------------------------------


def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram


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
class Light_Pass_Filter2_Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.feature_extractor = build_network(opt['network_g'])
        self.feature_extractor = self.model_to_device(self.feature_extractor)

        # # load pretrained models
        # load_path = self.opt['path'].get('pretrain_network_hq', None)
        # logger = get_root_logger()
        # if load_path is not None:
        #     logger.info(f'Loading net_g from {load_path}')
        #     self.load_network(self.feature_extractor, load_path, self.opt['path']['strict_load'])

        self.init_training_settings()
        self.all_vector_gram_layer0 = []
        self.all_vector_gram_layer1 = []
        self.all_light_factor_layer0 = []
        self.all_light_factor_layer1 = []

        self.labels0 = []
        self.labels1 = []


    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']

        self.feat_weights = {'layer0':0.5, 'layer1':0.5}

        lr_fpf1 = 1e-3 
        lr_fpf2 = 1e-3
        self.LightPassFilter1 = LightPassFilter_conv1(32896) # 8256
        self.LightPassFilter1_optimizer = torch.optim.Adamax([p for p in self.LightPassFilter1.parameters() if p.requires_grad == True], lr=lr_fpf1)
        self.LightPassFilter1.to(self.device)
        load_path = self.opt['path'].get('pretrain_network_LightPassFilter1', None)
        if load_path is not None:
            logger.info(f'Loading LightPassFilter1 from {load_path}')
            self.load_network(self.LightPassFilter1, load_path, self.opt['path']['strict_load'])

        # self.compress = nn.Conv2d(512, 256, 1, 1, 1)
        self.LightPassFilter2 = LightPassFilter_res1(131328)
        self.LightPassFilter2_optimizer = torch.optim.Adamax([p for p in self.LightPassFilter2.parameters() if p.requires_grad == True], lr=lr_fpf2)
        self.LightPassFilter2.to(self.device)
        load_path = self.opt['path'].get('pretrain_network_LightPassFilter2', None)
        if load_path is not None:
            logger.info(f'Loading LightPassFilter2 from {load_path}')
            self.load_network(self.LightPassFilter2, load_path, self.opt['path']['strict_load'])


        self.lightpassfilter_loss = losses.ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=0.1,
        distance=CosineSimilarity(),
        reducer=MeanReducer()
        )

        # set up optimizers and schedulers
        self.setup_schedulers()


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


    def optimize_parameters(self, current_iter):

        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.LightPassFilter1.parameters():
            p.requires_grad = True
        for p in self.LightPassFilter2.parameters():
            p.requires_grad = True

        loss_dict = OrderedDict()


        gt_feat_dict = self.feature_extractor(self.gt)
        lq_feat_dict = self.feature_extractor(self.lq)

        gt_feat0 = gt_feat_dict['dark2'].detach()
        gt_feat1 = gt_feat_dict['dark3'].detach()
        self.gt_feats = {'layer0': gt_feat0, 'layer1':gt_feat1}

        lq_feat0 = lq_feat_dict['dark2'].detach()
        lq_feat1 = lq_feat_dict['dark3'].detach()
        self.lq_feats = {'layer0': lq_feat0, 'layer1': lq_feat1}

        total_lf_loss = 0.0

        for idx, layer in enumerate(self.feat_weights):
            gt_feat = self.gt_feats[layer]
            lq_feat = self.lq_feats[layer]

            light_pass_filter_loss = 0.0

            if idx == 0:
                LightPassFilter = self.LightPassFilter1
                LightPassFilter_optimizer = self.LightPassFilter1_optimizer
            elif idx==1:
                LightPassFilter = self.LightPassFilter2
                LightPassFilter_optimizer = self.LightPassFilter2_optimizer

            LightPassFilter.train()
            LightPassFilter_optimizer.zero_grad()


            batch_size,_,_,_ = gt_feat.size()
            lq_gram = [0]*batch_size
            gt_gram = [0]*batch_size
            vector_lq_gram = [0]*batch_size
            vector_gt_gram = [0]*batch_size
            light_factor_lq = [0]*batch_size
            light_factor_gt = [0]*batch_size

            # 循环操作每张 image
            for batch_idx in range(batch_size):
                lq_gram[batch_idx] = gram_matrix(lq_feat[batch_idx])
                gt_gram[batch_idx] = gram_matrix(gt_feat[batch_idx])

                # 从给定的batch中的Gram矩阵提取上三角（不包括对角线）元素，并将这些元素转换为可训练的变量
                vector_lq_gram[batch_idx] = Variable(lq_gram[batch_idx][torch.triu(torch.ones(lq_gram[batch_idx].size()[0], lq_gram[batch_idx].size()[1])) == 1], requires_grad=True)
                vector_gt_gram[batch_idx] = Variable(gt_gram[batch_idx][torch.triu(torch.ones(gt_gram[batch_idx].size()[0], gt_gram[batch_idx].size()[1])) == 1], requires_grad=True)

                # t1 = vector_lq_gram[batch_idx]
                # t2 = vector_gt_gram[batch_idx]
                # print(f'vector_lq_gram[batch_idx]: {t1.shape}')
                # print(f'vector_gt_gram[batch_idx]: {t2.shape}')
                light_factor_lq[batch_idx] = LightPassFilter(vector_lq_gram[batch_idx])
                light_factor_gt[batch_idx] = LightPassFilter(vector_gt_gram[batch_idx])

            factor_list = []

            for batch_idx in range(batch_size):
                factor_list.append(torch.unsqueeze(light_factor_lq[batch_idx],0))
                factor_list.append(torch.unsqueeze(light_factor_gt[batch_idx],0))

            light_factor_embeddings = torch.cat(factor_list, 0)

            light_factor_embeddings_norm = torch.norm(light_factor_embeddings, p=2, dim=1).detach()
            size_light_factor = light_factor_embeddings.size()
            light_factor_embeddings = light_factor_embeddings.div(light_factor_embeddings_norm.expand(size_light_factor[1],batch_size*2).t())
            if batch_size == 4:
                light_factor_labels = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1])
            elif batch_size == 8:
                light_factor_labels = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            elif batch_size == 16: 
                light_factor_labels = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

            light_pass_filter_loss = self.lightpassfilter_loss(light_factor_embeddings,light_factor_labels)
            total_lf_loss +=  light_pass_filter_loss 
            
            tn = f'{layer}_light_pass_filter_loss'
            loss_dict[tn] = light_pass_filter_loss
            loss_dict['total_lf_loss'] = total_lf_loss

        total_lf_loss.backward(retain_graph=False)

        self.LightPassFilter1_optimizer.step()
        self.LightPassFilter2_optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.light_pass_filter_loss_dict = loss_dict


    def test(self):

        self.feature_extractor.eval()
        
        gt_feat_dict = self.feature_extractor(self.gt)
        lq_feat_dict = self.feature_extractor(self.lq)

        gt_feat0 = gt_feat_dict['stem'].detach()
        gt_feat1 = gt_feat_dict['dark2'].detach()
        self.gt_feats = {'layer0': gt_feat0, 'layer1':gt_feat1}

        lq_feat0 = lq_feat_dict['stem'].detach()
        lq_feat1 = lq_feat_dict['dark2'].detach()
        self.lq_feats = {'layer0': lq_feat0, 'layer1': lq_feat1}

        total_lf_loss = 0.0

        for idx, layer in enumerate(self.feat_weights):
            gt_feat = self.gt_feats[layer]
            lq_feat = self.lq_feats[layer]

            light_pass_filter_loss = 0.0

            if idx == 0:
                LightPassFilter = self.LightPassFilter1
                LightPassFilter_optimizer = self.LightPassFilter1_optimizer
            elif idx==1:
                LightPassFilter = self.LightPassFilter2
                LightPassFilter_optimizer = self.LightPassFilter2_optimizer

            LightPassFilter.eval()
            LightPassFilter_optimizer.zero_grad()

            self.lq_img_feat = lq_feat
            batch_size,_,_,_ = lq_feat.size()
            lq_gram = [0]*batch_size
            self.vector_lq_gram = [0]*batch_size
            self.light_factor_lq = [0]*batch_size

            self.gt_img_feat = gt_feat
            gt_gram = [0]*batch_size
            self.vector_gt_gram = [0]*batch_size
            self.light_factor_gt = [0]*batch_size

            # 循环操作每张 image
            for batch_idx in range(batch_size):

                lq_gram[batch_idx] = gram_matrix(lq_feat[batch_idx])
                # 从给定的batch中的Gram矩阵提取上三角（不包括对角线）元素，并将这些元素转换为可训练的变量
                self.vector_lq_gram[batch_idx] = lq_gram[batch_idx][torch.triu(torch.ones(lq_gram[batch_idx].size()[0], lq_gram[batch_idx].size()[1])) == 1]
                self.light_factor_lq[batch_idx] = LightPassFilter(self.vector_lq_gram[batch_idx])

                gt_gram[batch_idx] = gram_matrix(gt_feat[batch_idx])
                # 从给定的batch中的Gram矩阵提取上三角（不包括对角线）元素，并将这些元素转换为可训练的变量
                self.vector_gt_gram[batch_idx] = gt_gram[batch_idx][torch.triu(torch.ones(gt_gram[batch_idx].size()[0], gt_gram[batch_idx].size()[1])) == 1]
                self.light_factor_gt[batch_idx] = LightPassFilter(self.vector_gt_gram[batch_idx])

                f1 = self.vector_gt_gram[batch_idx].view(batch_size, -1).cpu().detach().numpy()
                f2 = self.light_factor_gt[batch_idx].view(batch_size, -1).cpu().detach().numpy()
                f3 = self.vector_lq_gram[batch_idx].view(batch_size, -1).cpu().detach().numpy()
                f4 = self.light_factor_lq[batch_idx].view(batch_size, -1).cpu().detach().numpy()

                if layer == 'layer0':
                    self.all_vector_gram_layer0.append(f1)
                    self.all_vector_gram_layer0.append(f3)

                    self.all_light_factor_layer0.append(f2)
                    self.all_light_factor_layer0.append(f4)
                    self.labels0.extend([0,1])

                else:
                    self.all_vector_gram_layer1.append(f1)
                    self.all_vector_gram_layer1.append(f3)

                    self.all_light_factor_layer1.append(f2)
                    self.all_light_factor_layer1.append(f4)
                    self.labels1.extend([0,1])


    def validation(self, dataloader, current_iter, tb_logger):

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')


        for idx, val_data in enumerate(dataloader):

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()


    def visual_feature(self, dataloader, current_iter, tb_logger):

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

            # if idx == 10:break


        pbar.close()

        ## 绘制 tSNE
        self.draw_tSNE(np.concatenate(self.all_vector_gram_layer0), 'all_vector_gram_at_layer0')
        self.draw_tSNE(np.concatenate(self.all_vector_gram_layer1), 'all_vector_gram_at_layer1')

        self.draw_tSNE(np.concatenate(self.all_light_factor_layer0), 'all_light_factor_at_layer0')
        self.draw_tSNE(np.concatenate(self.all_light_factor_layer1), 'all_light_factor_at_layer1')


    def draw_tSNE(self, feature, name):

        # print(f'feature: {feature.shape}')

        tsne = TSNE(n_components=2, perplexity=20, n_iter=1000)
        # tsne = TSNE(n_components=2)
        features_tsne = tsne.fit_transform(feature)

        # 二分类

        print(f'label1: {len(self.labels1)}')
        print(f'label1: {(self.labels1)}')

        print(f'feature: {len(feature)}')

        plt.figure(figsize=(10, 6))

        colors = ['red', 'blue']
        for i, c in enumerate(colors):
            # 使用布尔索引来选择属于同一类的数据点
            mask = np.array(self.labels1) == i
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], c=color, label=f'Class {i}')

            # plt.scatter(features_tsne[np.array(self.labels1) == i][:, 0], features_tsne[np.array(self.labels1) == i][:, 1], c=c, label=f'Class {i}')

        output_dir = 't_SNE'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # plt.scatter(features_tsne[:, 0], features_tsne[:, 1])
        # plt.colorbar()
        plt.title(f't-SNE Visualization of {name}')
        plt.xlabel('t-SNE axis 1')
        plt.ylabel('t-SNE axis 2')

        plt.savefig(f'{output_dir}/{name}.png')
        plt.show()
        print(f'draw tSNE:{name}')


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


    def save(self, epoch, current_iter):
        self.save_network(self.LightPassFilter1, 'LightPassFilter1', current_iter)
        self.save_network(self.LightPassFilter2, 'LightPassFilter2', current_iter)
        self.save_training_state(epoch, current_iter)



