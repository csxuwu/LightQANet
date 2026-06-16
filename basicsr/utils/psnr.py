

import cv2
import numpy as np
import math
# from tools import ops


def array_mu(array):
	'''
	计算数组均值
	:param array:
	:return:
	'''
	sum = 0
	for i in range(len(array)):
		sum += array[i]
	return sum / len(array)

def psnr(img1,img2):
	'''
	计算psnr
	:param img1:从网络得到的图像，为tensor类型
	:param img2:
	:return:
	'''
	psnr = []
	for i in range(len(img1)):
		img1_cpu,img2_cpu = img1.cpu(),img2.cpu()
		img1_np = img1_cpu[i].detach().numpy()
		img2_np = img2_cpu[i].detach().numpy()

		mse = np.mean((img1_np/1.0 - img2_np/1.0)**2)
		# if mse < 1.0e-10:
		#     return 100
		psnr_val = 10 * math.log10((1.0 ** 2)/mse)       # 此时img1、img2是经过归一化后的，即像素值在[0,1]
		psnr.append(psnr_val)
	return array_mu(psnr)

def psrn2(img1,img2):
	mse = ()