import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import json
from tps.utils import project_root
import imageio
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg') 
import random

class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output+1e-8, target.long())
        loss = torch.mean(loss * pixelWiseWeight)
        return loss


def oneMix(cfg, mask, data = None, target = None):
    #Mix
    if cfg.SOURCE == 'Viper':
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 190, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 0, 0, 142, 0, 0, 70,
                       0, 60, 100, 0, 0, 230, 119, 11, 32]
    elif cfg.SOURCE == 'SynthiaSeq':
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    def colorize_mask(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask


    if not (data is None):
        stackedMask0 = torch.broadcast_tensors(mask.float(), data[0])[0]
        
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)

    if not (target is None):
        stackedMask0 = torch.broadcast_tensors(mask.float(), target[0])[0]
        
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1])

    return data, target


def generate_class_mask(pred, classes):  #  [h, w]  [n]

    pred, classes = torch.broadcast_tensors(pred.squeeze().unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))  # [1, h, w]  [n, 1, 1] -- > pred: [n, h, w]  classes: [n, h, w]
    
    N = pred.eq(classes.float()).sum(0)
    return N 


def Class_mix(cfg, image1, image2, label1, label2, mask_img, mask_lbl, cls_mixer, cls_list, x1=None, y1=None, ch=None, cw=None, patch_re=True, path_list=None, sam_14 = None):
    if cfg.SOURCE == 'Viper':
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 190, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 0, 0, 142, 0, 0, 70,
                       0, 60, 100, 0, 0, 230, 119, 11, 32]
    elif cfg.SOURCE == 'SynthiaSeq':
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    def colorize_mask(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask

    inputs_, _ = oneMix(cfg, mask_img, data=torch.cat((image1, image2)))
    _, targets_ = oneMix(cfg, mask_lbl, target=torch.cat((label1, label2)))
    
    if patch_re == True:
        inputs_, targets_, path_list, Masks_longtail = cls_mixer.mix(cfg, inputs_.squeeze(0), cls_list, x1, y1, ch, cw, patch_re, in_lbl=targets_, path_list=None, sam_14 = None)
        return inputs_, targets_.unsqueeze(0), path_list, Masks_longtail
    else:
        inputs_, targets_ = cls_mixer.mix(cfg, inputs_.squeeze(0), cls_list, x1, y1, ch, cw, patch_re, in_lbl=targets_, path_list=path_list, sam_14 = None)
        return inputs_, targets_.unsqueeze(0)
    


def Class_mix_nolongtail(cfg, image1, image2, label1, label2, mask_lbl):

    inputs_, _ = oneMix(cfg, mask_lbl, data=torch.cat((image1, image2)))
    _, targets_ = oneMix(cfg, mask_lbl, target=torch.cat((label1, label2)))
    
    return inputs_, targets_


def Class_mix_flow(cfg, mask_flow=None, src_flow_last_cd=None, trg_flow=None):

    mixed_flow, _ = oneMix(cfg, mask_flow, data = torch.cat((src_flow_last_cd.float(),trg_flow.float())))
    return mixed_flow


class rand_mixer():
    def __init__(self, dataset, device):
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.device = device
        self.dataset = dataset
        if self.dataset == "viper_seq":
            jpath = str(project_root / 'data/viper_ids2path.json')
            self.class_map = {3: 0, 4: 1, 9: 2, 11: 3, 13: 4, 14: 5, 7: 6, 8: 6, 6: 7, 2: 8, 20: 9, 24: 10, 27: 11,
                          26: 12, 23: 13, 22: 14}
            self.ignore_ego_vehicle = True
            self.root = str(project_root / 'data/Viper')
            self.image_size = (1280, 720)
            self.labels_size = self.image_size
        elif self.dataset == "synthia_seq":
            jpath = str(project_root / 'data/synthia_ids2path.json')
            self.class_map = {3: 0, 4: 1, 2: 2, 5: 3, 7: 4, 15: 5, 9: 6, 6: 7, 1: 8, 10: 9, 11: 10, 8: 11,}
            self.ignore_ego_vehicle = False
            self.root = str(project_root / 'data/Cityscapes')
            self.image_size = (1280, 760)
            self.labels_size = self.image_size
        else:
            print('rand_mixer {} unsupported'.format(self.dataset))
            return
        
        with open(jpath, 'r') as load_f:
            self.ids2img_dict = json.load(load_f)


    def get_image(self, file):
        return self._load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return self._load_img(file, self.labels_size, Image.NEAREST, rgb=False)

    def get_labels_synthia_seq(self, file):
        lbl = imageio.imread(file, format='PNG-FI')[:, :, 0]
        img = Image.fromarray(lbl)
        img = img.resize(self.labels_size, Image.NEAREST)
        return np.asarray(img, np.float32)

    def _load_img(self, file, size, interpolation, rgb):
        
        img = Image.open(file)
        if rgb:
            img = img.convert('RGB')
        img = img.resize(size, interpolation)
        return np.asarray(img, np.float32)


    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def mix(self, cfg, in_img, classes, x1, y1, ch, cw, patch_re, in_lbl=None, path_list=None, sam_14 = None):
        a_c = 0

        if patch_re:
            path_list=[]
            if cfg.SOURCE == 'Viper' and sam_14:
                classes.append(14)
        classes = np.unique(classes)
        if patch_re == True:
            Masks_longtail = torch.zeros_like(in_lbl).cuda(self.device)

        cls_num_ = 0
        for i in classes:
            cls_num_ += 1
            if patch_re == True:
                while(True):
                    name = random.sample(self.ids2img_dict[str(i)], 1)
                    
                    if self.dataset == "viper_seq":
                        img_path = os.path.join(cfg.DATA_DIRECTORY_SOURCE, 'train/img' , name[0])

                        image = self.get_image(img_path)
                        
                        label_path = os.path.join(cfg.DATA_DIRECTORY_SOURCE, 'train/cls' , name[0].replace('jpg','png'))
                        label = self.get_labels(label_path)
                        if self.ignore_ego_vehicle:
                            lbl_car = label == 24
                            ret, lbs, stats, centroid = cv2.connectedComponentsWithStats(np.uint8(lbl_car))
                            lb_vg = lbs[-1, lbs.shape[1] // 2]
                            if lb_vg > 0:
                                label[lbs == lb_vg] = 0

                    elif self.dataset == "synthia_seq":
                        img_path = os.path.join(cfg.DATA_DIRECTORY_SOURCE, 'rgb', name[0])
                        image = self.get_image(img_path)[:-120, :, :]

                        label_path = os.path.join(cfg.DATA_DIRECTORY_SOURCE, 'label', name[0])
                        label = self.get_labels_synthia_seq(label_path)[:-120, :]
                        
                    label_copy = 255 * np.ones(label.shape, dtype=np.float32)
                    for k, v in self.class_map.items():
                        label_copy[label == k] = v

                    image = self.preprocess(image)
                    if i ==14:
                        if torch.sum(generate_class_mask(torch.Tensor(label_copy.copy())[x1:x1+cw, y1:y1+ch], torch.Tensor([i]).type(torch.int64))) > 100 :
                            break
                    else:
                        if torch.sum(generate_class_mask(torch.Tensor(label_copy.copy())[x1:x1+cw, y1:y1+ch], torch.Tensor([i]).type(torch.int64))) > 5 :
                            break

                img = torch.Tensor(image.copy()).unsqueeze(0).cuda(self.device)
                lbl = torch.Tensor(label_copy.copy()).cuda(self.device)
                
                path_list.append(img)
                path_list.append(lbl)
               
            elif patch_re == False:
                
                img = path_list[a_c]
                a_c += 1
                lbl = path_list[a_c]
                a_c += 1

            class_i = torch.Tensor([i]).type(torch.int64).cuda(self.device)
            MixMask = generate_class_mask(lbl[x1:x1+cw, y1:y1+ch], class_i).cuda(self.device)
            
            if patch_re == True:
                Masks_longtail += MixMask.float()
           

            if cls_num_ == 1:
                mixdata = torch.cat((img[:,:,x1:x1+cw, y1:y1+ch], in_img.unsqueeze(0).cuda(self.device)))
            elif cls_num_>1:
                mixdata = torch.cat((img[:,:,x1:x1+cw, y1:y1+ch], data.cuda(self.device)))

            if in_lbl is None:
                data, _ = oneMix(cfg, MixMask, data=mixdata)
               
            else:
               
                if cls_num_ == 1:
                    mixtarget = torch.cat((lbl[x1:x1+cw, y1:y1+ch].unsqueeze(0), in_lbl.cuda(self.device)), 0)
                elif cls_num_>1:
                    mixtarget = torch.cat((lbl[x1:x1+cw, y1:y1+ch].unsqueeze(0), target.unsqueeze(0).cuda(self.device)), 0)
                    
                data, target = oneMix(cfg, MixMask.float(), data=mixdata, target=mixtarget)

              
        if in_lbl is None:
            return data
        else:
            if patch_re:
                return data, target, path_list, Masks_longtail
            else:
                return data, target