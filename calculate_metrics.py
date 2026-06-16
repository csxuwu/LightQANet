import argparse
import cv2
import numpy as np
from os import path as osp

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.utils import scandir, get_root_logger, get_time_str
from basicsr.utils.matlab_functions import bgr2ycbcr
import logging
from glob import glob
import lpips

import torch
import torch.nn as nn
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

class LPIPSLoss(nn.Module):
    def __init__(self,
            loss_weight=1.0,
            use_input_norm=False,
            range_norm=False,):
        super(LPIPSLoss, self).__init__()
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False).eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.range_norm:
            pred   = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred   = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return lpips_loss.mean()



def main(args):
    """Calculate PSNR and SSIM for images.
    """

    print('-'*50)
    print(f'gt: {args.gt}')
    print(f'restored: {args.restored}')
    print('-'*50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = LPIPSLoss().to(device)

    psnr_all = []
    ssim_all = []
    lpips_all = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(args.restored, recursive=True, full_path=True)))
    

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')
    bs_psnr = 0
    for img_path_gt in img_list_gt:
        basename, ext = osp.splitext(osp.basename(img_path_gt))
        img_gt = cv2.imread(img_path_gt, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if 'SMG-LLIE-main' in args.root_log:
            img_gt = cv2.resize(img_gt, dsize=(512, 512))
        elif 'QuadPrior' in args.root_log:
            img_gt = cv2.resize(img_gt, dsize=(256, 256))
            
        if args.name == 'NASA':
            basename = basename.replace('-rtx00','')
            basename = basename+'.png'
        else:
            # 有些方法给图像的命名不同，计算时，需要调整
            # basename = basename.replace('normal', '')
            # basename = basename + '_' + basename + '_40000'
            # basename = basename+'.jpg'
            # basename = basename+'.png'
            # 自动匹配 restored 中与 basename 匹配的文件（支持多后缀）
            possible_exts = ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']
            found = False
            for ext in possible_exts:
                # candidate_path = osp.join(args.restored, basename + '_15000' + ext)
                candidate_path = osp.join(args.restored, basename + ext)
                print(f'candidate_path: {candidate_path}')
                if osp.exists(candidate_path):
                    # basename = basename + '_15000' + ext
                    basename = basename + ext

                    found = True
                    break
            if not found:
                print(f"❌ 找不到与 {basename} 匹配的图像，请检查路径或后缀")
                continue  # 跳过该图像


        # 拼接 restored 图像路径
        img_path_restored = osp.join(args.restored, basename)
        
        logger.info(f'gt: {osp.basename(img_path_gt)}')
        logger.info(f'rs: {osp.basename(img_path_restored)}')

        print(f'gt: {osp.basename(img_path_gt)}')
        print(f'rs: {osp.basename(img_path_restored)}')

        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # 如果尺寸不一致，则将 gt resize 到 restored 的尺寸
        if img_restored.shape[:2] != img_gt.shape[:2]:
            h, w = img_restored.shape[:2]
            img_gt = cv2.resize(img_gt, (w, h), interpolation=cv2.INTER_LINEAR)


        # --- LPIPS 计算 ---
        # 转为 Tensor，格式为 [1, 3, H, W]
        img_gt_tensor = torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).to(device)
        img_rs_tensor = torch.from_numpy(img_restored.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            lpips_score = lpips_fn(img_rs_tensor, img_gt_tensor).item()
        lpips_all.append(lpips_score)

        # logger.info(f'{basename:25}. \tLPIPS: {lpips_score:.6f}')


        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        if img_gt.shape != img_restored.shape:
            print('resize img_restored')
            img_restored = cv2.resize(img_restored, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        logger.info(f'{basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}, \tLPIPS: {lpips_score:.6f}')
        psnr_all.append(psnr)
        ssim_all.append(ssim)

        if psnr > bs_psnr:
            bs_psnr = psnr
            print('-'*30)
            print(f'best psnr: {basename}: {bs_psnr}')
            print('-'*30)

    print(f'end best psnr: {basename}: {bs_psnr}')
    # ...existing code...

    print(args.gt)
    print(args.restored)
    # print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')
    logger.info(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}, LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='LOL_v2_Real', help='testset name')
    parser.add_argument('--gt', type=str, default='/data/xuwu/datasets/low_light/LOL_v2/Real_captured/Test/Normal', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored_set', type=str, default='/data/xuwu/codes/low_light/PromptEnhance/results/LLIE_Prior_PromptV2_5_Arch_LOLv2Real_All_20240811_105650/visualization/LOLv2_Real_Dataset_Normal_rec', help='Path to restored images')
    parser.add_argument('--root_log', type=str, default=r'/data/xuwu/codes/low_light/PromptEnhance/results/LLIE_Prior_PromptV2_5_Arch_LOLv2Real_All_20240811_105650', help='Crop border for each side')

    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        # action='store_true',
        type=bool,
        default=False,
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', 
        # action='store_true', 
        default=False,
        help='Correct the mean and var of restored images.')
    args = parser.parse_args()

    # args.restored = osp.join(args.root_log, args.restored_set)
    args.restored = args.root_log

    log_file = osp.join(args.root_log, f"{args.restored_set}_{get_time_str()}.log")

    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)


    main(args)









