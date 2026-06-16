import logging
from os import path as osp
import os
# 这行代码必须放在 os、torch 之间
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path, opt_path, is_sh):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, opt_path, is_train=False, is_sh=is_sh)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # opt['datasets']['test']['dataroot'] = os.path.join(opt['datasets']['test']['dataroot'], opt['datasets']['test']['name'])

    # mkdir and initialize loggers
    make_exp_dirs(opt)

    log_file_name =  f"{opt['name']}_{get_time_str()}.log"
    log_path = opt['path']['log']
    log_file = f'{log_path}/{log_file_name}'


    # log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    # 可以测试多个测试集
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        # model.validation(test_loader, current_iter=opt['name'], tb_logger=None)
        model.visual_feature(test_loader, current_iter=opt['name'], tb_logger=None)


if __name__ == '__main__':
    # root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    torch.set_num_threads(2)  # 限制 cpu 线程数
    is_sh = False
    root_path = r'/data/xuwu/codes/low_light/PromptEnhance'
    opt_path = r'/data/xuwu/codes/low_light/PromptEnhance/options/LPF_test.yaml'
    # opt_path = r'/home/wuxu/codes/PromptEnhance/PromptEnhance_Codes/options/exploring3/test_ops/FiveK.yml'
    test_pipeline(root_path, opt_path, is_sh)







