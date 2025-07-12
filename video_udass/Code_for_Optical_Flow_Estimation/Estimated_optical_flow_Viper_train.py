import os 
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
# from models.sdc_net2d import *
from models.sdc_net2d_OF import *
import flow_vis

parser = argparse.ArgumentParser()
parser.add_argument('--resize_ratio',  default=1.0, type=float, help='Image resize ratio')
parser.add_argument('--pretrained', default='', type=str, metavar='PAT_imgH', help='path to trained video reconstruction checkpoint')
parser.add_argument('--flownet2_checkpoint', default='', type=str, metavar='PAT_imgH', help='path to flownet-2 best checkpoint')
parser.add_argument('--source_dir', default='', type=str, help='directory for data (default: Cityscapes root directory)')
parser.add_argument('--target_dir', default='', type=str, help='directory to save augmented data')
parser.add_argument('--sequence_length', default=2, type=int, metavar="SEQUENCE_LENGT_imgH",
                    help='number of interpolated frames (default : 2)')
parser.add_argument("--rgb_max", type=float, default = 255.)
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--propagate', type=int, default=1, help='propagate how many steps')
parser.add_argument('--vis', action='store_true', default=False, help='augment color encoded segmentation map')

def main():
	global args
	args = parser.parse_args()

	net = get_model()
	net.eval()
	model = net.cuda()

	if not os.path.exists(args.target_dir):
		os.makedirs(args.target_dir)

	list_path = 'DA-VSN/davsn/dataset/Viper_list/train.txt'
	with open(list_path) as f:
		file_names = [i_id.strip() for i_id in f]

	prop = args.propagate
	save_dir = args.target_dir
	with torch.no_grad():
		for ind, file_name in enumerate(file_names):
			frame = int(file_name.replace('.jpg', '')[-5:])
			names = []; imgs = []
			for k in range(frame-prop-1, frame+1):
				name = file_name.replace(str(frame).zfill(5) + '.jpg', str(k).zfill(5) + '.jpg')
				if k == frame:
					img_dir = os.path.join(args.source_dir, 'train', name)
				else:
					img_dir = os.path.join(args.source_dir, 'train', name)
				img = cv2.imread(img_dir)
				if args.resize_ratio != 1:
					height, width = img.shape[:2]
					img = cv2.resize(img, (round(args.resize_ratio*width), round(args.resize_ratio*height)), interpolation=cv2.INTER_CUBIC)
				imgs.append(img)
				name = name.split('/')[-1]
				names.append(name)

			file_name = file_name.split('/')[-1]
			print(str(ind),'/', str(len(file_names)) ,'---', file_name)

			img1_rgb = T_img(imgs[0]); img2_rgb = T_img(imgs[1]); img3_rgb = T_img(imgs[2])
			input_dict = {}; input_dict['image'] = [img1_rgb, img2_rgb, img3_rgb]
			flow_img1, pred_img1, _ = model(input_dict)

			pred_img1_name = file_name.replace('.jpg', str(frame-1).zfill(5) + '.jpg')
			pred_img1_np = (pred_img1.data.cpu().numpy().squeeze().transpose(1, 2, 0)).astype(np.uint8)
			pred_img_name = file_name.replace('.jpg', str(frame).zfill(5) + '.jpg')
			pred_img_np = imgs[2]
			cv2.imwrite(os.path.join(save_dir, pred_img1_name), pred_img1_np)
			cv2.imwrite(os.path.join(save_dir, pred_img_name), pred_img_np)

			flow_img1_np = flow_img1.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
			flow_color_img1 = flow_vis.flow_to_color(flow_img1_np, convert_to_bgr=False)
			flow_color_img1_name = file_name.replace('.jpg', str(frame-1).zfill(5) + '_flow_color.jpg')
			cv2.imwrite(os.path.join(save_dir, flow_color_img1_name), flow_color_img1)

			flow_img1_int16_x10 = np.int16(flow_img1_np*10)
			flow_img1_int16_name = file_name.replace('.jpg', str(frame-1).zfill(5) + '_int16_x10')
			np.save(os.path.join(save_dir, flow_img1_int16_name), flow_img1_int16_x10)

			# import pdb
			# pdb.set_trace()


def T_img(img):
	img_rgb = img.transpose((2,0,1))
	img_rgb = np.expand_dims(img_rgb, axis=0)
	img_rgb = torch.from_numpy(img_rgb.astype(np.float32))
	img_rgb = Variable(img_rgb).contiguous().cuda()
	return img_rgb

def T_label(img):
	img = np.expand_dims(img, axis=0)
	img = np.expand_dims(img, axis=0)
	img = torch.from_numpy(img.astype(np.float32))
	img = Variable(img).contiguous().cuda()
	return img

def get_model():
	model = SDCNet2DRecon(args)
	checkpoint = torch.load(args.pretrained)
	args.start_epoch = 0 if 'epoch' not in checkpoint else checkpoint['epoch']
	state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
	model.load_state_dict(state_dict, strict=False)
	print("Loaded checkpoint '{}' (at epoch {})".format(args.pretrained, args.start_epoch))
	return model

if __name__ == '__main__':
	main()

