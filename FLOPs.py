


import torch
from thop import profile
from thop import clever_format

import os

# 这行代码必须放在 os、torch 之间
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import datetime
import logging
import math
import time
import shutil
from os import path as osp
# from torchstat import stat
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
torch.backends.cudnn.benchmark = True

import warnings
# ignore UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
warnings.filterwarnings("ignore", category=UserWarning)


def FLOPs(model):
    # 假设 'model' 是你的PyTorch模型，'input' 是一个正确维度的输入张量
    # input = torch.randn(1,3,224,224)
    # input = input.to(model.device)

    # 计算模型的FLOPs和参数数量
    flops, params = profile(model.net_g, inputs=(input, ), verbose=False)
    trainable_params = sum(p.numel() for p in model.net_g.parameters() if p.requires_grad)

    # 打印出美化后的FLOPs和参数数量
    flops, params, trainable_params = clever_format([flops,params, trainable_params], "%.3f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    print(f"Trainable Params: {trainable_params}")


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        new_name = new_name.replace('tb_logger', 'tb_logger_archived')
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        shutil.move(path, new_name)
    os.makedirs(path, exist_ok=True)


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger

 
def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}'
                                                         f''
                                                         f''
                                                         f'')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path, opt_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, opt_path=opt_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            # os.makedirs(osp.join(opt['root_path'], 'tb_logger_archived'), exist_ok=True)
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'], opt)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    model.print_network(model.net_g)

    # FLOPs(model)
    


if __name__ == '__main__':
    # BASICSR_JIT = True
    # os.environ['BASICSR_JIT'] = True
    os.environ['RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'

    torch.set_num_threads(2)  # 限制 cpu 线程数

    # root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    root_path = r'/home/wuxu/codes/promptenhance/'
    # opt_path = r'/home/wuxu/codes/RIDCP/options/LL_SS_Syn5_wx3.yml'
    # opt_path = r'/home/wuxu/codes/RIDCP/options/exploring3/LOL/LLIE_prior_OS_Refer21_LOLv2_syn_wx.yml'
    # opt_path = r'/home/wuxu/codes/RIDCP/options/exploring3/Syn5_ops/LLIE_prior_OS_Refer21_exp3_Syn5_wx.yml'
    # opt_path = r'/home/wuxu/codes/RIDCP/options/exploring3/AGLLNet_ops/LLIE_prior_OS_Refer2_exp3_AGLLNet_wx.yml'
    # opt_path = r'/home/wuxu/codes/RIDCP/options/exploring3/LOL/LLIE_prior_OS_Refer21_VE_LOL_L_Syn_wx.yml'
    # opt_path_list = [
    # # '/home/wuxu/codes/RIDCP/options/exploring3/LSRW/LLIE_prior_OS_Refer21_LSRW_Huawei3_wx.yml',
    # # '/home/wuxu/codes/RIDCP/options/exploring3/LSRW/LLIE_prior_OS_Refer21_LSRW_Huawei4_wx.yml',
    # '/home/wuxu/codes/RIDCP/options/exploring3/LSRW/LLIE_prior_OS_Refer21_LSRW_Huawei5_wx.yml',
    # '/home/wuxu/codes/RIDCP/options/exploring3/LSRW/LLIE_prior_OS_Refer21_LSRW_Huawei6_wx.yml'
    # ]
    
    # for opt_path in opt_path_list:

    #     train_pipeline(root_path, opt_path)

    opt_path = r'/home/wuxu/codes/promptenhance/options/exploring3/FLOPs.yaml'
    train_pipeline(root_path, opt_path)
