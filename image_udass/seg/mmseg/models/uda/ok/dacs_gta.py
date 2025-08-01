# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.dacs_transforms_zz import (get_class_masks_quad_s, get_class_masks_quad_t, color_jitter, gaussian_blur)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio
from tools import transformsgpu
import PIL
import json
from PIL import Image, ImageOps
from tools import transformmasks
IMG_MEAN = np.array((123.675, 116.28, 103.53), dtype=np.float32)
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import gc
import torch.cuda as cuda
import imageio

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


def strongTransform_ammend(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    device = data.device
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def strongTransform_class_mix(image1, image2, label1, label2, mask_img, mask_lbl, cls_mixer, cls_list, strong_parameters, local_iter, vis = False, note = None):
    inputs_, _ = transformsgpu.oneMix(mask_img, data=torch.cat((image1.unsqueeze(0), image2.unsqueeze(0))))
   
    _, targets_ = transformsgpu.oneMix(mask_lbl, target=torch.cat((label1.unsqueeze(0), label2.unsqueeze(0))))

    inputs_, targets_ = cls_mixer.mix(inputs_.squeeze(0), targets_.squeeze(0), cls_list, local_iter, vis, note)
    targets_ = targets_.unsqueeze(0)

    inputs_, targets_ = color_jitter(
        color_jitter=strong_parameters['color_jitter'],
        s=strong_parameters['color_jitter_s'],
        p=strong_parameters['color_jitter_p'],
        mean=strong_parameters['mean'],
        std=strong_parameters['std'],
        data=inputs_,
        target=targets_)
    inputs_, targets_ = gaussian_blur(blur=strong_parameters['blur'], data=inputs_, target=targets_)

    return inputs_, targets_


def strong_Trans_form_(param, label1=None, label2=None):
    device = label1.device
   
    _, target = transformsgpu.oneMix(param['mix'], target=torch.cat((label1.unsqueeze(0), label2.unsqueeze(0))))
    target = target.squeeze(0) 

    _, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=None,
        target=target)
    _, target = gaussian_blur(blur=param['blur'], data=None, target=target)
    
    return target

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'gta':
        return './data/gta'


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        img, mask = Image.fromarray(img, mode="RGB"), Image.fromarray(mask, mode="L")
        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask, dtype=np.uint8)


class RandomCrop_gta(object):  # used for results in the CVPR-19 submission
    def __init__(self, size, padding=0):
       
        self.size = tuple(size)
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size

        x1 = random.randint(0, int(w) - tw)
        y1 = random.randint(0, int(h) - th)

        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class rand_mixer():
    def __init__(self, root, dataset):
        if dataset == "gta5":
            jpath = './data/gta5_ids2path.json'
            self.resize = (2048, 1024) #(1280, 720) 
            input_size = (1024, 1024) #(640, 640)
            self.data_aug = Compose([RandomCrop_gta(input_size)])
        elif dataset == "cityscapes":
            jpath = './data/cityscapes_ids2path.json'
        else:
            print('rand_mixer {} unsupported'.format(dataset))
            return
        self.root = root
        self.dataset = dataset
        self.class_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                     26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        with open(jpath, 'r') as load_f:
            self.ids2img_dict = json.load(load_f)

    def mix(self, in_img, in_lbl, classes, local_iter, vis, note):
        img_size = in_lbl.shape
        device = in_img.device
        cls_num_ = 0
        for i in classes:
            cls_num_ += 1
            if self.dataset == "gta5":
                while(True):
                    try:
                        name = random.sample(self.ids2img_dict[str(i)], 1)
                        img_path = os.path.join(self.root, "images/%s" % name[0])
                        label_path = os.path.join(self.root, "labels/%s" % name[0])
                        img = Image.open(img_path)
                        lbl = Image.open(label_path)
                        img = img.resize(self.resize, Image.Resampling.BILINEAR) #BICUBIC
                        lbl = lbl.resize(self.resize, Image.Resampling.NEAREST)
                        img = np.array(img, dtype=np.uint8)
                        lbl = np.array(lbl, dtype=np.uint8)
                        img, lbl = self.data_aug(img, lbl) # random crop to input_size
                        img = np.asarray(img, np.float32)
                        lbl = np.asarray(lbl, np.float32)
                        label_copy = 255 * np.ones(lbl.shape, dtype=np.float32)
                    except:
                        name = random.sample(self.ids2img_dict[str(i)], 1)
                        img_path = os.path.join(self.root, "images/%s" % name[0])
                        label_path = os.path.join(self.root, "labels/%s" % name[0])
                        img = Image.open(img_path)
                        lbl = Image.open(label_path)
                        img = img.resize(self.resize, Image.Resampling.BILINEAR) #BICUBIC
                        lbl = lbl.resize(self.resize, Image.Resampling.NEAREST)
                        img = np.array(img, dtype=np.uint8)
                        lbl = np.array(lbl, dtype=np.uint8)
                        img, lbl = self.data_aug(img, lbl) # random crop to input_size
                        img = np.asarray(img, np.float32)
                        lbl = np.asarray(lbl, np.float32)
                        label_copy = 255 * np.ones(lbl.shape, dtype=np.float32)

                    for k, v in self.class_map.items():
                        label_copy[lbl == k] = v
                    if i in label_copy:
                        lbl = label_copy.copy()
                        img = img[:, :, ::-1].copy()  # change to BGR
                        img -= IMG_MEAN
                        img = img.transpose((2, 0, 1)) / 255.
                        if torch.sum(transformmasks.generate_class_mask(torch.Tensor(lbl), torch.Tensor([i]).type(torch.int64))) > 50:
                            break
                img = torch.Tensor(img).cuda(device)
                lbl = torch.Tensor(lbl).cuda(device)
                class_i = torch.Tensor([i]).type(torch.int64).cuda(device)
                MixMask = transformmasks.generate_class_mask(lbl, class_i)

                if cls_num_ == 1:
                   
                    mixdata = torch.cat((img.unsqueeze(0), in_img ))
                    
                    mixtarget = torch.cat((lbl.unsqueeze(0), in_lbl.squeeze(0)))

                elif cls_num_>1:
                    
                    mixdata = torch.cat((img.unsqueeze(0), data))
                    mixtarget = torch.cat((lbl.unsqueeze(0), target))



                data, target = transformsgpu.oneMix(MixMask.float(), data=mixdata, target=mixtarget)

        return data, target


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 4000
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)

        self.weak_img_tmpl = torch.zeros(2, 3, 640, 640)
        self.weak_target_img_tmpl = torch.zeros(2, 3, 640, 640)
        self.gt_semantic_seg_tmpl = torch.zeros(2, 1, 640, 640)
        self.pseudo_label_tmpl = torch.zeros(2, 1, 640, 640)
        self.pseudo_logits_tmpl = torch.zeros(2, 1, 640, 640)
        self.gt_pixel_weight_tmpl = torch.zeros(2, 640, 640)
        self.pseudo_weight_tmpl = torch.zeros(2, 640, 640)

        self.writer = SummaryWriter(log_dir='./logs_former/')

        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        if mask is not None:
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
        
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 4000:
            self._init_ema_weights()

        if self.local_iter > 4000:
            self._update_ema(self.local_iter)
          
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        no_parameters = {
            'mix': None,
            'color_jitter': 0,
            'color_jitter_s': 0,
            'color_jitter_p': 0,
            'blur': 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        weak_img, weak_target_img = img.clone(), target_img.clone()
        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        if not self.source_only:
            # Generate pseudo-label
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img, target_img_metas)
            seg_debug['Target'] = self.get_ema_model().debug_output

            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
                ema_logits)
            ps_large_logits = torch.max(torch.softmax(ema_logits.detach(), dim=1), dim=1)[0].ge(0.9).long() == 1
            del ema_logits

            pseudo_weight = self.filter_valid_pseudo_region(
                pseudo_weight, valid_pseudo_mask)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            if self.local_iter < 6000:
                # Apply mixing
                mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
                mixed_seg_weight = pseudo_weight.clone()
                mix_masks = get_class_masks(gt_semantic_seg)

                for i in range(batch_size):
                    strong_parameters['mix'] = mix_masks[i]
                    mixed_img[i], mixed_lbl[i] = strong_transform(
                        strong_parameters,
                        data=torch.stack((img[i], target_img[i])),
                        target=torch.stack(
                            (gt_semantic_seg[i][0], pseudo_label[i])))
                    _, mixed_seg_weight[i] = strong_transform(
                        strong_parameters,
                        target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
                # del gt_pixel_weight
                mixed_img = torch.cat(mixed_img)
                mixed_lbl = torch.cat(mixed_lbl)

                # Train on mixed images
                mix_losses = self.get_model().forward_train(
                    mixed_img,
                    img_metas,
                    mixed_lbl,
                    seg_weight=mixed_seg_weight,
                    return_feat=False,
                )
                seg_debug['Mix'] = self.get_model().debug_output
                mix_losses = add_prefix(mix_losses, 'mix')
                mix_loss, mix_log_vars = self._parse_losses(mix_losses)
                log_vars.update(mix_log_vars)
                mix_loss.backward()
            
            elif self.local_iter >= 6000:
                    # Apply quad-mixing
                # S2S
                mixed_img_ss, mixed_lbl_ss = [None] * batch_size, [None] * batch_size
                mix_masks_tmpl, classes_s = get_class_masks_quad_s(self.gt_semantic_seg_tmpl)
                pseudo_weight_ss  = [None] * batch_size

                for i in range(batch_size):
                    no_parameters['mix'] = mix_masks_tmpl[i]
                    mixed_img_ss[i], mixed_lbl_ss[i] = strong_transform(
                        no_parameters,
                        data=torch.stack((self.weak_img_tmpl[i], weak_img[i])),
                        target=torch.stack((self.gt_semantic_seg_tmpl[i][0], gt_semantic_seg[i][0])))
                    _, pseudo_weight_ss[i] = strong_transform(
                        no_parameters,
                        target=torch.stack((self.gt_pixel_weight_tmpl[i], gt_pixel_weight[i])))
                mixed_img_ss = torch.cat(mixed_img_ss)
                mixed_lbl_ss = torch.cat(mixed_lbl_ss).squeeze()
                pseudo_weight_ss = torch.cat(pseudo_weight_ss).squeeze()


                # S2T
                mixed_img_st, mixed_lbl_st = [None] * batch_size, [None] * batch_size
                pseudo_weight_st  = [None] * batch_size
                for i in range(batch_size):
                    no_parameters['mix'] = mix_masks_tmpl[i]
                    mixed_img_st[i], mixed_lbl_st[i] = strong_transform(
                        no_parameters,
                        data=torch.stack((self.weak_img_tmpl[i], weak_target_img[i])),
                        target=torch.stack((self.gt_semantic_seg_tmpl[i][0], pseudo_label[i])))
                    _, pseudo_weight_st[i] = strong_transform(
                        no_parameters,
                        target=torch.stack((self.gt_pixel_weight_tmpl[i], pseudo_weight[i])))
                    
                mixed_img_st = torch.cat(mixed_img_st)
                mixed_lbl_st = torch.cat(mixed_lbl_st).squeeze()
                pseudo_weight_st = torch.cat(pseudo_weight_st).squeeze()

                if self.local_iter >= 11000 and self.local_iter < 11020:
                    m = torch.tensor(self.gt_pixel_weight_tmpl[0]).unsqueeze(0)
                    m = torch.cat([m,m,m],dim=0).cpu().numpy()
                    plt.figure()
                    plt.imshow((m * 255).transpose(1,2,0).astype(np.uint8))
                    plt.savefig('./keshihua_former/gt_pixel_weight_tmpl'+str(self.local_iter)+'.png')
                    plt.close()

                    m = torch.tensor(pseudo_weight[0]).unsqueeze(0)
                    m = torch.cat([m,m,m],dim=0).cpu().numpy()
                    plt.figure()
                    plt.imshow((m * 255).transpose(1,2,0).astype(np.uint8))
                    plt.savefig('./keshihua_former/pseudo_weight'+str(self.local_iter)+'.png')
                    plt.close()

                    m = torch.tensor(pseudo_weight_st[0]).unsqueeze(0)
                    m = torch.cat([m,m,m],dim=0).cpu().numpy()
                    plt.figure()
                    plt.imshow((m * 255).transpose(1,2,0).astype(np.uint8))
                    plt.savefig('./keshihua_former/pseudo_weight_st_'+str(self.local_iter)+'.png')
                    plt.close()

                # target_tmpl
                gta5_cls_mixer = rand_mixer(get_data_path('gta'), "gta5")
                class_to_select = [3, 4, 5, 6, 12, 15, 16, 17, 18]

                cls_to_use = random.sample(class_to_select, 2)
                # T2SS
                
                mix_masks_tmpl_t = get_class_masks_quad_t(self.pseudo_label_tmpl, self.pseudo_logits_tmpl, classes_s)

                mixed_img_tss0, mixed_lbl_tss0 = strongTransform_class_mix(self.weak_target_img_tmpl[0], mixed_img_ss[0], self.pseudo_label_tmpl[0],
                                                                    mixed_lbl_ss[0], mix_masks_tmpl_t[0], mix_masks_tmpl_t[0], gta5_cls_mixer, cls_to_use,
                                                                    strong_parameters, self.local_iter, vis = False, note = 'tss')
                mixed_img_tss1, mixed_lbl_tss1 = strongTransform_class_mix(self.weak_target_img_tmpl[1], mixed_img_ss[1], self.pseudo_label_tmpl[1],
                                                                    mixed_lbl_ss[1], mix_masks_tmpl_t[1], mix_masks_tmpl_t[1], gta5_cls_mixer, cls_to_use,
                                                                    strong_parameters, self.local_iter)
                strong_parameters['mix'] = mix_masks_tmpl_t[0]
            
                pseudo_weight0 = strong_Trans_form_(strong_parameters, self.pseudo_weight_tmpl[0], pseudo_weight_ss[0])
                strong_parameters['mix'] = mix_masks_tmpl_t[1]
                pseudo_weight1 = strong_Trans_form_(strong_parameters, self.pseudo_weight_tmpl[1], pseudo_weight_ss[1])

                mixed_img_tss = torch.cat((mixed_img_tss0, mixed_img_tss1))
                mixed_lbl_tss = torch.cat((mixed_lbl_tss0, mixed_lbl_tss1)) 
                pseudo_weight_tss = torch.cat((pseudo_weight0, pseudo_weight1)).squeeze()

                # T2ST
                strong_parameters['mix'] = None

                mixed_img_tst0, mixed_lbl_tst0 = strongTransform_class_mix(self.weak_target_img_tmpl[0], mixed_img_st[0], self.pseudo_label_tmpl[0],
                                                                    mixed_lbl_st[0], mix_masks_tmpl_t[0], mix_masks_tmpl_t[0], gta5_cls_mixer, cls_to_use,
                                                                    strong_parameters, self.local_iter, vis = False, note = 'tst')
                mixed_img_tst1, mixed_lbl_tst1 = strongTransform_class_mix(self.weak_target_img_tmpl[1], mixed_img_st[1], self.pseudo_label_tmpl[1],
                                                                    mixed_lbl_st[1], mix_masks_tmpl_t[1], mix_masks_tmpl_t[1], gta5_cls_mixer, cls_to_use,
                                                                    strong_parameters, self.local_iter)
                strong_parameters['mix'] = mix_masks_tmpl_t[0]
                pseudo_weight0 = strong_Trans_form_(strong_parameters, self.pseudo_weight_tmpl[0], pseudo_weight_st[0])
                strong_parameters['mix'] = mix_masks_tmpl_t[1]
                pseudo_weight1 = strong_Trans_form_(strong_parameters, self.pseudo_weight_tmpl[1], pseudo_weight_st[1])

                # if self.local_iter >= 11000 and self.local_iter <  11020:
                #     m = torch.tensor(self.pseudo_weight_tmpl[0]).unsqueeze(0)
                #     m = torch.cat([m,m,m],dim=0).cpu().numpy()
                #     plt.figure()
                #     plt.imshow((m * 255).transpose(1,2,0).astype(np.uint8))
                #     plt.savefig('./keshihua_former/pseudo_weight_tmpl'+str(self.local_iter)+'.png')
                #     plt.close()

                #     m = torch.tensor(pseudo_weight_st[0]).unsqueeze(0)
                #     # print('m.shape', m.shape)
                #     m = torch.cat([m,m,m],dim=0).cpu().numpy()
                #     # print('m.shape', m.shape)
                #     plt.figure()
                #     plt.imshow((m * 255).transpose(1,2,0).astype(np.uint8))
                #     plt.savefig('./keshihua_former/pseudo_weight_st'+str(self.local_iter)+'.png')
                #     plt.close()

                #     m = torch.tensor(pseudo_weight0).squeeze(0)
                #     # print('m.shape', m.shape)
                #     m = torch.cat([m,m,m],dim=0).cpu().numpy()
                #     # print('m.shape', m.shape)
                #     plt.figure()
                #     plt.imshow((m * 255).transpose(1,2,0).astype(np.uint8))
                #     plt.savefig('./keshihua_former/pseudo_weight_tst_'+str(self.local_iter)+'.png')
                #     plt.close()


                mixed_img_tst = torch.cat((mixed_img_tst0, mixed_img_tst1))
                mixed_lbl_tst = torch.cat((mixed_lbl_tst0, mixed_lbl_tst1)) 
                pseudo_weight_tst = torch.cat((pseudo_weight0, pseudo_weight1)).squeeze()

                # Train on mixed images
                mix_losses_tss = self.get_model().forward_train(mixed_img_tss, img_metas, mixed_lbl_tss.long(), seg_weight=pseudo_weight_tss,
                                                            return_feat=False)
                # Train on mixed images
                 
                seg_debug['mix_tss'] = self.get_model().debug_output
                mix_losses_tss = add_prefix(mix_losses_tss, 'mix_tss')
                mix_tss_loss, mix_tss_log_vars = self._parse_losses(mix_losses_tss)
                log_vars.update(mix_tss_log_vars)
                mix_tss_loss = mix_tss_loss *0.5
                mix_tss_loss.backward()

                mix_tst_losses = self.get_model().forward_train(mixed_img_tst, img_metas, mixed_lbl_tst.long(), seg_weight=pseudo_weight_tst,
                                                            return_feat=False)
                seg_debug['mix_tst'] = self.get_model().debug_output
                mix_tst_losses = add_prefix(mix_tst_losses, 'mix_tst')
                mix_tst_loss, mix_tst_log_vars = self._parse_losses(mix_tst_losses)
                log_vars.update(mix_tst_log_vars)
                mix_tst_loss = mix_tst_loss *0.5
                mix_tst_loss.backward()

            # if self.local_iter >= 11000 and self.local_iter <  11020:
            #     inputs_ss_0 = self.weak_img_tmpl[0].squeeze().permute(1, 2, 0)
            #     plt.figure()
            #     plt.imshow((inputs_ss_0.cpu().numpy()).astype(np.uint8))
            #     plt.savefig('./keshihua_former/weak_img_tmpl'+ str(self.local_iter)+ '.png')
            #     plt.close()

            #     inputs_ss_0 = weak_img[0].squeeze().permute(1, 2, 0)
            #     plt.figure()
            #     plt.imshow((inputs_ss_0.cpu().numpy()).astype(np.uint8))
            #     plt.savefig('./keshihua_former/weak_img'+ str(self.local_iter)+ '.png')
            #     plt.close()

            #     inputs_ss_0 = mixed_img_ss[0].squeeze().permute(1, 2, 0)
            #     plt.figure()
            #     plt.imshow((inputs_ss_0.cpu().numpy()).astype(np.uint8))
            #     plt.savefig('./keshihua_former/mixed_img_ss'+ str(self.local_iter)+ '.png')
            #     plt.close()

            #     inputs_ss_0 = weak_target_img[0].squeeze().permute(1, 2, 0)
            #     plt.figure()
            #     plt.imshow((inputs_ss_0.cpu().numpy()).astype(np.uint8))
            #     plt.savefig('./keshihua_former/weak_target_img'+ str(self.local_iter)+ '.png')
            #     plt.close()

            #     inputs_ss_0 = mixed_img_st[0].squeeze().permute(1, 2, 0)
            #     plt.figure()
            #     plt.imshow((inputs_ss_0.cpu().numpy()).astype(np.uint8))
            #     plt.savefig('./keshihua_former/mixed_img_st'+ str(self.local_iter)+ '.png')
            #     plt.close()

            #     inputs_ss_0 = self.weak_target_img_tmpl[0].squeeze().permute(1, 2, 0)
            #     plt.figure()
            #     plt.imshow((inputs_ss_0.cpu().numpy()).astype(np.uint8))
            #     plt.savefig('./keshihua_former/weak_target_img_tmpl'+ str(self.local_iter)+ '.png')
            #     plt.close()

            #     inputs_ss_0 = mixed_img_tss0.squeeze().permute(1, 2, 0)
            #     plt.figure()
            #     plt.imshow((inputs_ss_0.cpu().numpy()).astype(np.uint8))
            #     plt.savefig('./keshihua_former/mixed_img_tss'+ str(self.local_iter)+ '.png')
            #     plt.close()

            #     inputs_ss_0 = mixed_img_tst0.squeeze().permute(1, 2, 0)
            #     plt.figure()
            #     plt.imshow((inputs_ss_0.cpu().numpy()).astype(np.uint8))
            #     plt.savefig('./keshihua_former/mixed_img_tst'+ str(self.local_iter)+ '.png')
            #     plt.close()

              

            #     sou_labels_mix_0 = self.gt_semantic_seg_tmpl[0][0].cpu().squeeze()
            #     amax_output_col = colorize_mask(np.asarray(sou_labels_mix_0, dtype=np.uint8))
            #     amax_output_col.save('./keshihua_former/gt_semantic_seg_tmpl'+str(self.local_iter)+'.png')

            #     sou_labels_mix_0 = gt_semantic_seg[0][0].cpu().squeeze()
            #     amax_output_col = colorize_mask(np.asarray(sou_labels_mix_0, dtype=np.uint8))
            #     amax_output_col.save('./keshihua_former/gt_semantic_seg'+str(self.local_iter)+'.png')

            #     sou_labels_mix_0 = mixed_lbl_ss[0].cpu().squeeze()
            #     amax_output_col = colorize_mask(np.asarray(sou_labels_mix_0, dtype=np.uint8))
            #     amax_output_col.save('./keshihua_former/mixed_lbl_ss'+str(self.local_iter)+'.png')

            #     sou_labels_mix_0 = pseudo_label[0].cpu().squeeze()
            #     amax_output_col = colorize_mask(np.asarray(sou_labels_mix_0, dtype=np.uint8))
            #     amax_output_col.save('./keshihua_former/pseudo_label'+str(self.local_iter)+'.png')

            #     sou_labels_mix_0 = mixed_lbl_st[0].cpu().squeeze()
            #     amax_output_col = colorize_mask(np.asarray(sou_labels_mix_0, dtype=np.uint8))
            #     amax_output_col.save('./keshihua_former/mixed_lbl_st'+str(self.local_iter)+'.png')

            #     sou_labels_mix_0 = self.pseudo_label_tmpl[0].cpu().squeeze()
            #     amax_output_col = colorize_mask(np.asarray(sou_labels_mix_0, dtype=np.uint8))
            #     amax_output_col.save('./keshihua_former/pseudo_label_tmpl'+str(self.local_iter)+'.png')

            #     sou_labels_mix_0 = mixed_lbl_tss0.cpu().squeeze()
            #     amax_output_col = colorize_mask(np.asarray(sou_labels_mix_0, dtype=np.uint8))
            #     amax_output_col.save('./keshihua_former/mixed_lbl_tss'+str(self.local_iter)+'.png')

            #     sou_labels_mix_0 = mixed_lbl_tst0.cpu().squeeze()
            #     amax_output_col = colorize_mask(np.asarray(sou_labels_mix_0, dtype=np.uint8))
            #     amax_output_col.save('./keshihua_former/mixed_lbl_tst'+str(self.local_iter)+'.png')

            #     m = torch.tensor(mix_masks_tmpl[0].squeeze(0))
            #     # print('m.shape', m.shape)
            #     m = torch.cat([m,m,m],dim=0).cpu().numpy()
            #     # print('m.shape', m.shape)
            #     plt.figure()
            #     plt.imshow((m * 255).transpose(1,2,0).astype(np.uint8))
            #     plt.savefig('./keshihua_former/mix_masks_tmpl'+str(self.local_iter)+'.png')
            #     plt.close()

            #     m = torch.tensor(mix_masks_tmpl_t[0].squeeze(0))
            #     # print('m.shape', m.shape)
            #     m = torch.cat([m,m,m],dim=0).cpu().numpy()
            #     # print('m.shape', m.shape)
            #     plt.figure()
            #     plt.imshow((m * 255).transpose(1,2,0).astype(np.uint8))
            #     plt.savefig('./keshihua_former/mix_masks_tmpl_t'+str(self.local_iter)+'.png')
            #     plt.close()

        # Masked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask,
                                   pseudo_label, pseudo_weight)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()


        self.weak_img_tmpl = weak_img.detach().clone()
        self.weak_target_img_tmpl = weak_target_img.detach().clone()
        self.gt_semantic_seg_tmpl = gt_semantic_seg.detach().clone()
        self.pseudo_label_tmpl = pseudo_label.detach().clone()
        self.pseudo_logits_tmpl = ps_large_logits.detach().clone()
        self.gt_pixel_weight_tmpl = gt_pixel_weight.detach().clone()
        self.pseudo_weight_tmpl = pseudo_weight.detach().clone()

        if self.local_iter > 0 and self.local_iter < 6000:
            current_losses = { 'mix_loss': mix_loss, 'masked_loss': masked_loss}
            # print_losses(current_losses, self.local_iter)
            log_losses_tensorboard(self.writer, current_losses, self.local_iter)
 
        elif self.local_iter >= 6000:
            current_losses = { 'mix_loss': (mix_tss_loss + mix_tst_loss), 'mix_tst_loss': mix_tst_loss, 'mix_tss_loss': mix_tss_loss, 'masked_loss': masked_loss}
            log_losses_tensorboard(self.writer, current_losses, self.local_iter)
            # print_losses(current_losses, self.local_iter)

        self.local_iter += 1
        # print('self.local_iter', self.local_iter)
        return log_vars




def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()



palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 190, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 0, 0, 142, 0, 0, 70,
                   0, 60, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask