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


# -----------------------------------------
# 2024.05.01
# 基于LLIE_prior_OS_Refer2_exp3_model
# 将模型的返回用字典替代

# 添加了 domain discrimator
# domain discrimator 损失的细节
# 更新了 domain discrimator 的训练代码

# 添加了 light_pass filter , 直接用于计算 照度/风格之间的损失
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
class LLIE_Prior_Prompt3_Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # define network
        if opt.get('seg') is not None:
            opt['network_g']['seg_cfg'] = opt['seg']
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # print(opt['network_g']['weight_light'])

        # --------------------------------------------------------------
        # 加载预训练好的 decoder、codebook，并冻结参数
        # load pre-trained HQ ckpt, frozen decoder and codebook
        # --------------------------------------------------------------
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            # 用于生成 GT 量化下标
            hq_opt = self.opt['network_g'].copy()
            hq_opt['LQ_stage'] = False
            self.net_hq = build_network(hq_opt)
            self.net_hq = self.model_to_device(self.net_hq)
            self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            # VQ-GAN 第二阶段训练，加载StageI 的参数，并冻结部分模块的参数 ['quantize', 'decoder_group', 'after_quant_group', 'out_conv']
            self.load_network(self.net_g, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break
            else:
                # 指定可学习的模块
                learnable_module_keywords = self.opt['network_g'].get('learnable_module_keywords', None)
                if learnable_module_keywords is not None:
                    print(f'learnable_module_keywords: {learnable_module_keywords}')

                    for name, module in self.net_g.named_modules():
                        for p in module.parameters():
                            p.requires_grad = False

                    for name, module in self.net_g.named_modules():
                        for fkw in learnable_module_keywords:
                            if fkw in name:
                                for p in module.parameters():
                                    p.requires_grad = True
                                break
                        

        # --------------------------------------------------------------
        # 加载 stage I 训练好的参数，作为 stage II 的初始参数
        # load pretrained models
        # --------------------------------------------------------------
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
            self.net_d_best = copy.deepcopy(self.net_d)

        self.net_g_best = copy.deepcopy(self.net_g)
        # self.print_network(self.net_g)
        # self.print_network(self.net_g.context_module)


    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)

        # define domain Discriminator
        self.net_domain_d = build_network(self.opt['network_domain_d'])
        self.net_domain_d = self.model_to_device(self.net_domain_d)

        lr_fpf1 = 1e-3 
        lr_fpf2 = 1e-3
        self.LightPassFilter1 = LightPassFilter_conv1(8256) # 8256
        self.LightPassFilter1_optimizer = torch.optim.Adamax([p for p in self.LightPassFilter1.parameters() if p.requires_grad == True], lr=lr_fpf1)
        self.LightPassFilter1.to(self.device)

        self.LightPassFilter2 = LightPassFilter_res1(32896)
        self.LightPassFilter2_optimizer = torch.optim.Adamax([p for p in self.LightPassFilter2.parameters() if p.requires_grad == True], lr=lr_fpf2)
        self.LightPassFilter2.to(self.device)
        self.feat_weights = {'layer0':0.5, 'layer1':0.5}

        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        self.net_d.train()
        self.net_domain_d.train()

        # define losses
        self.cri_pix = None
        self.cri_perceptual = None
        self.cri_latent_contrast = None
        self.cri_content = None
        self.cri_gram_feat_contrast = None
        self.cri_clip_cos_img = None

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)

        if train_opt.get('latent_contrast_opt'):
            self.cri_latent_contrast = build_loss(train_opt['latent_contrast_opt']).to(self.device)

        if train_opt.get('content_opt'):
            if train_opt['content_opt']['type'] == 'l2':
                self.cri_content = nn.MSELoss()
            elif train_opt['content_opt']['type'] == 'Wasserstein':
                from scipy.stats import wasserstein_distance as wd
                self.cri_content = wd

        if train_opt.get('gram_feat_contrast_opt'):
            self.cri_gram_feat_contrast = build_loss(train_opt['gram_feat_contrast_opt']).to(self.device)
            
        if train_opt.get('domain_triplet_opt'):
            self.cri_triplet = build_loss(train_opt['domain_triplet_opt']).to(self.device)

        if train_opt.get('clip_cos_img_opt'):
            self.cri_clip_cos_img = build_loss(train_opt['clip_cos_img_opt']).to(self.device)

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.lightpassfilter_loss = losses.ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=0.1,
        distance=CosineSimilarity(),
        reducer=MeanReducer()
        )

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.feat_loss_weight = 1.0
        self.entropy_loss_weight = 0.5


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            # if not v.requires_grad:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

        # optimizer domain d
        optim_type = train_opt['optim_domain_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_domain_d = optim_class(self.net_domain_d.parameters(), **train_opt['optim_domain_d'])
        self.optimizers.append(self.optimizer_domain_d)


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


    def pre_optimize_light_filter(self, current_iter, train_light_filter_iter):

        for p in self.net_g.parameters():
            p.requires_grad = False
        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.LightPassFilter1.parameters():
            p.requires_grad = True
        for p in self.LightPassFilter2.parameters():
            p.requires_grad = True

        loss_dict = OrderedDict()

        if current_iter < train_light_filter_iter:
            _,_,_,_, gt_feat_dict = self.net_hq.encode_indices(self.gt)
            _,_,_,_, lq_feat_dict = self.net_hq.encode_indices(self.lq)
        else:
            gt_feat_dict = self.net_g.encoder(self.gt)
            lq_feat_dict = self.net_g.encoder(self.lq)

        gt_feat0 = gt_feat_dict['128'].detach()
        gt_feat1 = gt_feat_dict['64'].detach()
        self.gt_feats = {'layer0': gt_feat0, 'layer1':gt_feat1}

        lq_feat0 = lq_feat_dict['128'].detach()
        lq_feat1 = lq_feat_dict['64'].detach()
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

            # b = gt_feat.shape()[0]
            # print(b)
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

                light_factor_lq[batch_idx] = LightPassFilter(vector_lq_gram[batch_idx])
                light_factor_gt[batch_idx] = LightPassFilter(vector_gt_gram[batch_idx])

            light_factor_embeddings = torch.cat((
                    torch.unsqueeze(light_factor_lq[0],0),
                    torch.unsqueeze(light_factor_gt[0],0),

                    torch.unsqueeze(light_factor_lq[1],0),
                    torch.unsqueeze(light_factor_gt[1],0),

                    torch.unsqueeze(light_factor_lq[2],0),
                    torch.unsqueeze(light_factor_gt[2],0),

                    torch.unsqueeze(light_factor_lq[3],0),
                    torch.unsqueeze(light_factor_gt[3],0)),0)

            light_factor_embeddings_norm = torch.norm(light_factor_embeddings, p=2, dim=1).detach()
            size_light_factor = light_factor_embeddings.size()
            light_factor_embeddings = light_factor_embeddings.div(light_factor_embeddings_norm.expand(size_light_factor[1],8).t())
            light_factor_labels = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1])
            light_pass_filter_loss = self.lightpassfilter_loss(light_factor_embeddings,light_factor_labels)

            total_lf_loss +=  light_pass_filter_loss 
            
            tn = f'{layer}_light_pass_filter_loss'
            loss_dict[tn] = light_pass_filter_loss
            loss_dict['total_lf_loss'] = total_lf_loss
        total_lf_loss.backward(retain_graph=False)

        if current_iter < train_light_filter_iter:
            self.log_dict = self.reduce_loss_dict(loss_dict)
        self.light_pass_filter_loss_dict = loss_dict

    def optimize_parameters(self, current_iter, train_light_filter_iter):

        loss_dict = OrderedDict()
        train_opt = self.opt['train']

        ## 训练 light_filter
        self.pre_optimize_light_filter(current_iter, train_light_filter_iter)
        loss_dict.update(self.light_pass_filter_loss_dict)
        
        # --------------------------------------------------------------
        # Stage II 不训练 判别器
        # --------------------------------------------------------------
        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.net_g.parameters():
            p.requires_grad = True
        for p in self.LightPassFilter1.parameters():
            p.requires_grad = False
        for p in self.LightPassFilter2.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        # --------------------------------------------------------------
        # GT：获得GT 的 重构、量化特征、量化特征索引
        # lq：获得重构输出、损失
        # --------------------------------------------------------------
        with torch.no_grad():
            gt_indices, quant_gt, feat_to_quant_gt, after_quant_feat_gt, gt_feat_dict = self.net_hq.encode_indices(input=self.gt)
        self.lq.requires_grad = True
        self.out_dict = self.net_g(input=self.lq,
                                   gt_img=self.gt,
                                   reference_img=self.refer,
                                   gt_indices=gt_indices,
                                   net_hq=self.net_hq,
                                   prompt_text=self.gt_caption)
        self.output = self.out_dict['out_img']
        l_codebook = self.out_dict['codebook_loss']
        l_semantic = self.out_dict['semantic_loss']
        quant_g = self.out_dict['feat_to_quant']        # 对于LQ， feat_to_quant，相当于 quant
        quant_g_z = self.out_dict['z_quant']
        after_quant_feat_lq = self.out_dict['after_quant_feat']

        l_g_total = 0
        

        # ===================================================

        if 'transf_loss' in self.out_dict.keys():
            l_g_total += self.out_dict['transf_loss']
            loss_dict['transf_loss'] = self.out_dict['transf_loss']

        # =======================================================================
        if 'logits' in self.out_dict.keys():
            # 直接预测 code index
            # cross entropy
            pred_code_logits = self.out_dict['logits']
            # print('-'*100)
            # print(type(pred_code_logits))
            gt_indices = gt_indices[0].view(self.b, -1)
            cross_entropy_loss = F.cross_entropy(pred_code_logits.permute(0, 2, 1), gt_indices) * self.entropy_loss_weight
            l_g_total += cross_entropy_loss
            loss_dict['l_cross_entropy_loss'] = cross_entropy_loss

            # quant feat loss
            l_feat_encoder = torch.mean((quant_gt.detach() - quant_g) ** 2) * self.feat_loss_weight
            l_g_total += l_feat_encoder
            loss_dict['l_feat_encoder'] = l_feat_encoder


            if train_opt.get('quantization_opt', None):

                # quantization_alpha = 0.01
                quantization_loss = 0.0
                
                if train_opt['quantization_opt']['type'] == 'swdC':
                    quantization_loss = quantization_swdc_loss(pred_code_logits.contiguous().view(pred_code_logits.size(0), -1))
                
                l_g_total += quantization_loss * train_opt['quantization_opt']['loss_weight']
                loss_dict['l_quantization_loss'] = quantization_loss
        
        else:
            # 最近邻匹配得到 code index
            # codebook loss
            if train_opt.get('codebook_opt', None):
                l_codebook *= train_opt['codebook_opt']['loss_weight']
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()

        # =======================================================================
        # semantic cluster loss, only for LQ stage!
        if train_opt.get('semantic_opt', None) and isinstance(l_semantic, torch.Tensor):
            l_semantic *= train_opt['semantic_opt']['loss_weight']
            l_semantic = l_semantic.mean()
            l_g_total += l_semantic
            loss_dict['l_semantic'] = l_semantic
            
        # =======================================================================
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # =======================================================================
        # 隐变量对比损失
        if self.cri_latent_contrast:

            if train_opt['latent_contrast_opt'].get('type_encoder', None):
                if train_opt['latent_contrast_opt']['type_encoder'] == 'lq_encoder':
                    # 采用 lq encoder 来获得 特征
                    lq_encoder_out_gt = self.net_g.encoder(self.gt)
                    lq_encoder_out_rec = self.net_g.encoder(self.output)
                    lq_encoder_out_lq = self.net_g.encoder(self.lq)

                    l_la_cont = self.cri_latent_contrast(lq_encoder_out_rec['enc_feats_context'], lq_encoder_out_gt['enc_feats_context'], lq_encoder_out_lq['enc_feats_context'])

            else:

                with torch.no_grad():
                    _, quant_ouput, feat_to_quant_output, _ = self.net_hq.encode_indices(input=self.output)
                # l_la_cont = self.cri_latent_contrast(quant_ouput, quant_gt, quant_g_z)  # 计算的是 量化特征 之间的
                l_la_cont = self.cri_latent_contrast(feat_to_quant_output, feat_to_quant_gt, quant_g)

            l_la_cont *= train_opt['latent_contrast_opt']['loss_weight']
            l_g_total += l_la_cont
            loss_dict['l_latent_contrast'] = l_la_cont

        # =======================================================================
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style


        # =======================================================================
        if train_opt.get('domain_Loss_opt'):
            
            loss_type = train_opt['domain_Loss_opt']['type']
            if loss_type == 'BCE':
                criterion = nn.BCELoss()
            else:
                criterion = nn.BCELoss()    # 默认 采用 BCELoss

            ll_domain = 0       # 生成图像 fake
            gt_domain = 1       # 真实图像 real
            it = train_opt['domain_Loss_opt']['iter']

            if current_iter % it == 0:

                # -----------------
                # 训练 net_domain_d
                # -----------------
                
                for p in self.net_domain_d.parameters():
                    p.requires_grad = True
                self.optimizer_domain_d.zero_grad()

                ## 来自 真实图像（gt image）
                with torch.no_grad():   # 不更新生成器的参数
                    gt_feat = self.net_g.encoder(self.gt)
                    gt_feat = gt_feat['enc_feats'].detach()
                domain_d_out_gt = self.net_domain_d(gt_feat)
                real_domain_label = torch.FloatTensor(domain_d_out_gt.data.size()).fill_(gt_domain).to(self.device)

                loss_real = criterion(domain_d_out_gt, real_domain_label) / 2 
                loss_real.backward()
                loss_dict['l_domain_mse_gt_d'] = loss_real 

                ## 来自生成图像（ll image）
                with torch.no_grad():   # 不更新生成器的参数
                    lq_feat = self.net_g.encoder(self.lq)
                    lq_feat = lq_feat['enc_feats'].detach()
                domain_d_out_lq = self.net_domain_d(lq_feat)
                fake_domain_label = torch.FloatTensor(domain_d_out_lq.data.size()).fill_(ll_domain).to(self.device)
                loss_fake = criterion(domain_d_out_lq, fake_domain_label) / 2 

                loss_fake.backward()
                loss_dict['l_domain_mse_ll_d'] = loss_fake

                self.optimizer_domain_d.step()

            else:

                # -----------------
                # 训练 net_g
                # 此时 fake、real image的标签都为 true，即都为真实的
                # -----------------

                for p in self.net_domain_d.parameters():
                    p.requires_grad = False


                ## 来自生成图像（ll image）
                lq_feat = self.net_g.encoder(self.lq)
                lq_feat = lq_feat['enc_feats']
                domain_d_out_lq = self.net_domain_d(lq_feat)
                real_domain_label = torch.FloatTensor(domain_d_out_lq.data.size()).fill_(gt_domain).to(self.device)
                loss_gt = criterion(domain_d_out_lq, real_domain_label) / 2 * train_opt['domain_Loss_opt']['loss_weight']

                l_g_total += loss_gt
                loss_dict['l_domain_mse_gt_g'] = loss_gt

        # =======================================================================
        if train_opt.get('domain_GAN_opt'):
            
            ll_domain = 0
            gt_domain = 1
            
            it = train_opt['domain_GAN_opt']['iter']
            if current_iter % it == 0:
                # 训练 net_domain_d
                # gt 的标签为 real == 1 == True
                # ll 的标签为 fake == 0 == False
                for p in self.net_domain_d.parameters():
                    p.requires_grad = True
                self.optimizer_domain_d.zero_grad()

                # 来自 gt image 的图像/真实图像
                with torch.no_grad():
                    gt_feat = self.net_g.encoder(self.gt)
                    gt_feat = gt_feat['enc_feats'].detach()
                domain_d_out_gt = self.net_domain_d(F.softmax(gt_feat, dim=1))
                loss_ll2 = self.cri_gan(domain_d_out_gt, target_is_real=True, is_disc=True) 

                loss_ll2.backward()
                loss_dict['l_domain_GAN_ll_train_d'] = loss_ll2

                # 来自 ll image 的图像/生成图像
                with torch.no_grad():
                    lq_feat = self.net_g.encoder(self.lq)
                    lq_feat = lq_feat['enc_feats'].detach()
                domain_d_out_lq = self.net_domain_d(F.softmax(lq_feat, dim=1))
                loss_gt2 = self.cri_gan(domain_d_out_lq, target_is_real=False, is_disc=True) 

                loss_gt2.backward()
                loss_dict['l_domain_GAN_gt_train_d'] = loss_gt2

                self.optimizer_domain_d.step()

            else:
                # 训练 net_g
                # 标签都为 real == 1 == True
                for p in self.net_domain_d.parameters():
                    p.requires_grad = False

                ## 来自真实图像 （gt image）
                # gt_feat = self.net_g.encoder(self.gt)
                # gt_feat = gt_feat['enc_feats']
                # domain_d_out_gt = self.net_domain_d(gt_feat)

                # loss_ll2 = self.cri_gan(domain_d_out_gt, target_is_real=True, is_disc=False) * train_opt['domain_GAN_opt']['loss_weight']

                # l_g_total += loss_ll2
                # loss_dict['l_domain_GAN_ll_g'] = loss_ll2

                # 来自 ll image 的图像
                lq_feat = self.net_g.encoder(self.lq)
                lq_feat = lq_feat['enc_feats']
                domain_d_out_lq = self.net_domain_d(F.softmax(lq_feat, dim=1))
                loss_gt2 = self.cri_gan(domain_d_out_lq, target_is_real=True, is_disc=False) * train_opt['domain_GAN_opt']['loss_weight']

                l_g_total += loss_gt2
                loss_dict['l_domain_GAN_gt_g'] = loss_gt2

        # =======================================================================
        if train_opt.get('domain_triplet_opt'):

            triplet_loss = self.cri_triplet(batch=self.out_dict['feat_to_quant'], positive=feat_to_quant_gt)
            triplet_loss = train_opt['domain_triplet_opt']['loss_weight'] * triplet_loss
            l_g_total += triplet_loss


        # =======================================================================
        # 计算 light filter loss
        lq_feat0 = self.out_dict['enc_feat_dict']['128']
        lq_feat1 = self.out_dict['enc_feat_dict']['64']
        lq_feats = {'layer0': lq_feat0, 'layer1':lq_feat1}

        gt_feats_dict = self.net_g.encoder(self.gt)
        gt_feat0 = gt_feat_dict['128']
        gt_feat1 = gt_feat_dict['64']
        gt_feats = {'layer0': gt_feat0, 'layer1':gt_feat1}
        # fsm_weights = {'layer0':0.5, 'layer1':0.5}

        loss_fsm = 0.

        for idx, layer in enumerate(self.feat_weights):
            b_feature = lq_feats[layer]
            a_feature = gt_feats[layer]

            na,da,ha,wa = a_feature.size()
            nb,db,hb,wb = b_feature.size()

            if idx == 0:
                LightPassFilter = self.LightPassFilter1
                LightPassFilter_optimizer = self.LightPassFilter1_optimizer
            elif idx==1:
                LightPassFilter = self.LightPassFilter2
                LightPassFilter_optimizer = self.LightPassFilter2_optimizer
            
            LightPassFilter.eval()

            layer_fsm_loss = 0.
            for batch_idx in range(na):
                b_gram = gram_matrix(b_feature[batch_idx])
                a_gram = gram_matrix(a_feature[batch_idx])

                
                a_gram = a_gram *(hb*wb)/(ha*wa)

                # 获得 gram，且只包含上三角区域
                vector_b_gram = b_gram[torch.triu(torch.ones(b_gram.size()[0], b_gram.size()[1])).requires_grad_() == 1].requires_grad_()
                vector_a_gram = a_gram[torch.triu(torch.ones(a_gram.size()[0], a_gram.size()[1])).requires_grad_() == 1].requires_grad_()

                light_factor_b = LightPassFilter(vector_b_gram)
                light_factor_a = LightPassFilter(vector_a_gram)
                half = int(light_factor_b.shape[0]/2)
                        
                layer_fsm_loss += self.feat_weights[layer]*torch.mean((light_factor_b/(hb*wb) - light_factor_a/(ha*wa))**2)/half/ b_feature.size(0)
            loss_fsm += layer_fsm_loss / na

        light_pass_filter_loss = loss_fsm / (idx+1)
        l_g_total += light_pass_filter_loss
        loss_dict['l_light_pass_filter_loss'] =light_pass_filter_loss

        # =======================================================================
        if self.cri_clip_cos_img:
            # 计算两张图像的CLIP image feature之间的 余弦相似度
            l_clip_cos_img = self.cri_clip_cos_img(self.output, self.gt)
            l_g_total += l_clip_cos_img
            loss_dict['l_clip_cos_img'] = l_clip_cos_img
        # =======================================================================
        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = self.net_d(quant_g)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        # print(l_g_total.requires_grad)
        # if l_g_total.requires_grad:
        l_g_total.mean().backward()
        self.optimizer_g.step()

        ## optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(quant_gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(quant_g.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        # print(loss_dict.keys())
        # import sys
        # sys.exit()


    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
        lq_input = self.lq
        _, _, h, w = lq_input.shape 
        if h * w < min_size:
            # self.output,_ = net_g.test(self.gt)
            self.output = net_g.test(lq_input, reference=self.refer, net_hq=self.net_hq)
        else:
            self.output = net_g.test_tile(lq_input, reference=self.refer)
        self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None, visual_codebook=False):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir, visual_codebook=False):
        print(self.opt['path']['visualization'])
        dataset_name = dataloader.dataset.opt['name']
        # is_reference = dataloader.dataset.opt['is_reference']
        is_reference = dataloader.dataset.opt.get('is_reference', False)
        if is_reference:
            reference_image_root = dataloader.dataset.opt['reference_path']
            reference_image_list = glob.glob(os.path.join(reference_image_root, '*'))
        # print(len(reference_image_list))
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')

        for idx, val_data in enumerate(dataloader):

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if is_reference:
                _,_,h, w = self.lq.shape
                self.refer = random_load_images(path_list=reference_image_list, batchsize=self.b, size_h=h, size_w=w)
            else:
                self.refer = None

            self.test()

            sr_img = tensor2img(self.output)
            if not self.gt is None:
                gt_img = tensor2img(self.gt)

            # if visual_codebook:
            #     codebook_img = self.vis_single_code()
            #     save_codebook_path = osp.join(self.opt['path']['visualization'], img_name,
            #                                  f'{img_name}_codebook_{current_iter}.png')
            #     codebook_img = tensor2img(codebook_img)
            #     print(save_codebook_path)
            #     imwrite(codebook_img, save_codebook_path)

            #     del codebook_img

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                    save_img_path_gt = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_gt.png')
                    save_img_path_in = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_lq.png')


                    imwrite(sr_img, save_img_path)
                    # gt_img = tensor2img(self.gt)
                    in_img = tensor2img(self.lq)
                    imwrite(gt_img, save_img_path_gt)
                    imwrite(in_img, save_img_path_in)

                    # tentative for out of GPU memory
                    del self.gt
                    del self.lq
                    del self.output
                    torch.cuda.empty_cache()

                else:
                    weight_texture = self.opt['network_g']['weight_texture']
                    weight_style = self.opt['network_g']['weight_style']
                    sub_file_name = f'w1={weight_texture}_w2={weight_style}'
                    # opt['network_g']['weight_light']
                    if self.opt['network_g'].get('weight_light', None):
                        weight_light = self.opt['network_g']['weight_light']
                        sub_file_name = f'w1={weight_texture}_w2={weight_style}_w3=_{weight_light}'

                    if 'ExDark' in dataset_name:
                        subdir = dataloader.dataset.opt['subdir']
                        sub_file_name = f'w1={weight_texture}_w2={weight_style}/{subdir}'

                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 dataset_name,
                                                 sub_file_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 dataset_name,
                                                 sub_file_name,
                                                 f'{img_name}.png')
                        # f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_img_path)
                    del self.lq
                    del self.output

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            #         t = calculate_metric(metric_data, opt_)

            #         print(f'{name}: {t}')

            # print('-' * 50)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')



        pbar.close()

        # 加上下面这段代码，计算psnr时会出问题
        # for metric in self.metric_results.keys():
        #     self.metric_results[metric] /= (idx + 1)
        #     print(self.metric_results[metric])

        # print('-'*100)

        if with_metrics and self.opt['is_train']:
        # if with_metrics:

            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)



            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


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


    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx)
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)


    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        # if self.output != None:
        #     out_dict['result_codebook'] = self.output.detach().cpu()[:vis_samples]
        if self.output != None:
            out_dict['output'] = self.output.detach().cpu()[:vis_samples]
        if not self.LQ_stage:
            out_dict['codebook'] = self.vis_single_code()
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    # def init_CLIP(self):
    #     # CLIP
    #     self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
    #     for p in self.clip_model.parameters():
    #         p.requires_grad = False
    #     self.clip_model.eval()


    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)



