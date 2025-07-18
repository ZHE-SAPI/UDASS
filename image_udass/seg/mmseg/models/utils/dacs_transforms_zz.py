# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    # print('data.shape0', data.shape) #  [2, 3, 640, 640]
    # print('target.shape0', target.shape) #  [2, 1, 640, 640]

    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    # print('data.shape1', data.shape) #  [2, 3, 640, 640]
    # print('target.shape1', target.shape) #  [2, 1, 640, 640]

    return data, target









def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks

def get_class_masks_quad_s(labels):
    class_masks = []
    classes_s = []
    for label in labels:
        classes = torch.unique(label)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, 2, replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
        classes_s.append(class_choice)
    return class_masks, classes_s


def get_class_masks_quad_t(labels, pseudo_logits_tmpl, classes_s):
    class_masks = []
    # print('labels.shape', labels.shape)
    # print('classes_s', classes_s)
    # print('pseudo_logits_tmpl.shape', pseudo_logits_tmpl.shape)
    an_ = 0
    for label in labels:
        # print('label.shape', label.shape)
        # print('[an_]', an_)
        # print('classes_s[an_]', classes_s[an_])
        # print('classes_s[an_][0]', classes_s[an_][0])
        # print('classes_s[an_][1]', classes_s[an_][1])
        classes = torch.unique(label).float()
        b = torch.nonzero((classes != classes_s[an_][0]) * (classes != classes_s[an_][1] ), as_tuple=False).squeeze()
        classes = torch.index_select(classes, dim=0, index=b)
        nclasses_t = classes.shape[0]
        try:
            classes = (classes[torch.Tensor(np.random.choice(nclasses_t, 2 ,replace=False)).long()])
        except:
            try:
                classes = (classes[torch.Tensor(np.random.choice(nclasses_t, 1 ,replace=False)).long()])
            except:
                classes = torch.Tensor([729]).float()
        
        # class_choice = np.random.choice(
        #     nclasses, 2, replace=False)
        # classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0)* pseudo_logits_tmpl[an_])
        an_ += 1
    return class_masks



def get_class_masks_quad_t_quad(labels, pseudo_logits_tmpl):
    an_ = 0
    class_masks = []
    for label in labels:
        classes = torch.unique(labels).float()
        nclasses = classes.shape[0]
        try:
            class_choice = classes[torch.Tensor(np.random.choice(nclasses, 2, replace=False)).long()]
        except:
            try:
                class_choice = classes[torch.Tensor(np.random.choice(nclasses, 1, replace=False)).long()]
            except:
                classes = torch.Tensor([729]).float()
        
        # classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0)* pseudo_logits_tmpl[an_])
        an_ += 1
    return class_masks

def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
