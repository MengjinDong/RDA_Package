#!/usr/bin/env python

# run test on a simulated two pairs of images, and then compare the warped image with greedy output
import os
import random
import argparse
import time
import shutil
import csv
import numpy as np
import math
import importlib
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from functools import partial
import pickle
import matplotlib.pyplot as plt
import nibabel as nib

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import vxm_model as vxm  # nopep8
from vxm_model.py.utils import my_read_pytorch_image_from_nifti, my_read_pytorch_warp_from_nifti, single_sample_aug, get_moving_volume
from vxm_model.torch import layers

# first load the model to test

root_dir = "/data/mengjin"
vxm_dir = "/data/mengjin/Longi_T1_Aim2"
data_dir = "/data/mengjin/Longi_T1_2GO_QC/T1_Input_3d"
DEF_ROOT = "/data/mengjin/Longi_T1_2GO_QC/Longi_T1_Dataset_Deformation_Fields/"
load_attention = vxm_dir + "/vxm_attention_2_ALOHA_20231030/Model/2023-12-06_13-28/last_checkpoint_0024.pt"

gpus = "1".split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

model = vxm.networks.VxmSingleAttentionALOHA.load(load_attention, device)

# then load images

group = "MORE"
stage = "1"
# subject_fname = "018_S_4313_2012-08-01_2012-11-28_blmptrim_right_to_hw.nii.gz" # more 0
subject_fname = "006_S_5153_2013-08-02_2013-04-22_blmptrim_left_to_hw.nii.gz" # more 0
subject_fname = "018_S_4400_2011-12-20_2012-04-09_blmptrim_left_to_hw.nii.gz" # more 1

warp_name = subject_fname.replace('blmptrim_', "mp_antsreg3d_", 1).replace('_to_hw.nii.gz', '1Warp.nii.gz', 1)
warp_name = DEF_ROOT + "/" + warp_name

bl_name = f"{data_dir}/{group}/{stage}/{subject_fname}"
fu_name = bl_name.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)

save_fname = f"{vxm_dir}/vxm_attention_2_ALOHA_20231030/{subject_fname}"

bl_img, _ = my_read_pytorch_image_from_nifti(
    bl_name, dtype=torch.float32, device=device)
fu_img, _ = my_read_pytorch_image_from_nifti(
    fu_name, dtype=torch.float32, device=device)
warp, _ = my_read_pytorch_warp_from_nifti(
    warp_name, dtype=torch.float32, device=device)

# then feed images to the model

seg_img = bl_img

bl_img, \
fu_img, \
crop_pos,  \
segs,  \
warp = single_sample_aug(bl_img,
                        fu_img,
                        warp,
                        augment=['normalize', 'fixed_crop'],
                        add_batch_axis=False,
                        add_feat_axis=False,
                        pad_shape=None,
                        resize_factor=1,
                        segs=True,
                        vol_seg=seg_img,
                        max_angle=15,
                        rotate_prob=1, # 0.8
                        output_size=[48, 80, 64])
                        # output_size=output_size)

# warp = warp.permute(0, 4, 1, 2, 3)

x = torch.cat([bl_img, fu_img], dim=1)


out_seg = model.UNet3D_Seg(x)
out_seg, moved_img, volume_diff, volume, warped_volume, masks, warped_masks, jdet = model.attention(bl_img,
                                                                                fu_img, 
                                                                                warp,
                                                                                out_seg,
                                                                                registration=True)

volume_diff_divide_hippo, volume_diff_divide_sulcus = volume_diff
hippo_volume, sulcus_volume = volume
warped_hippo_volume, warped_sulcus_volume = warped_volume
hippo_mask, sulcus_mask = masks
warped_hippo_mask, warped_sulcus_mask = warped_masks

vol_diff_label = [volume_diff_divide_hippo, volume_diff_divide_sulcus]
attention_pred = torch.argmax(out_seg, dim=1)

# calculate this number by attention sum of voxels
hippo_attn_vox = (attention_pred == 1).sum(dim=(1, 2, 3))
sulcus_attn_vox = (attention_pred == 2).sum(dim=(1, 2, 3))

i = 0
            
row = [bl_name, fu_name,
        vol_diff_label[0][0].item(), vol_diff_label[1][0].item(),
        hippo_volume.item(), warped_hippo_volume.item(),
        sulcus_volume.item(), warped_sulcus_volume.item(),
        hippo_attn_vox.item(), sulcus_attn_vox.item(),
        ]

# print("vol_diff_label                                                 ", vol_diff_label[0][0].item(), vol_diff_label[1][0].item())
print("hippo_volume.item(), warped_hippo_volume.item()                ", hippo_volume.item(), warped_hippo_volume.item())
print("sulcus_volume.item(), warped_sulcus_volume.item()              ", sulcus_volume.item(), warped_sulcus_volume.item())
# print("hippo_attn_vox.item(), sulcus_attn_vox.item()                  ", hippo_attn_vox.item(), sulcus_attn_vox.item())

print("orig image1      volume", get_moving_volume(bl_img))
print("orig image2      volume", get_moving_volume(fu_img))
print("orig moved image volume", get_moving_volume(moved_img))
print(row)

# save moving image
bl_aug_name = save_fname.replace('.nii.gz', "_bl_aug.nii.gz", 1)
vxm.py.utils.save_volfile(bl_img[i, 0, :].detach().cpu().numpy().squeeze(), bl_aug_name)

# save fixed image
fu_aug_name = save_fname.replace('.nii.gz', "_fu_aug.nii.gz", 1)
vxm.py.utils.save_volfile(fu_img[i, 0, :].detach().cpu().numpy().squeeze(), fu_aug_name)

# save attention image
moving_attention_name = save_fname.replace('.nii.gz', "_attention_rda.nii.gz", 1)
vxm.py.utils.save_volfile(attention_pred[i, :].detach().cpu().numpy().squeeze().astype(float), moving_attention_name)

hippo_mask_name = save_fname.replace('.nii.gz', "_hippo_mask.nii.gz", 1)
vxm.py.utils.save_volfile(hippo_mask[i, :].detach().cpu().numpy().squeeze().astype(float), hippo_mask_name)

sulcus_mask_name = save_fname.replace('.nii.gz', "_sulcus_mask.nii.gz", 1)
vxm.py.utils.save_volfile(sulcus_mask[i, :].detach().cpu().numpy().squeeze().astype(float), sulcus_mask_name)

warped_hippo_mask_name = save_fname.replace('.nii.gz', "_warped_hippo_mask.nii.gz", 1)
vxm.py.utils.save_volfile(warped_hippo_mask[i, :].detach().cpu().numpy().squeeze().astype(float), warped_hippo_mask_name)

warped_sulcus_mask_name = save_fname.replace('.nii.gz', "_warped_sulcus_mask.nii.gz", 1)
vxm.py.utils.save_volfile(warped_sulcus_mask[i, :].detach().cpu().numpy().squeeze().astype(float), warped_sulcus_mask_name)

# save jacobian determinant
moving_jdet_name = save_fname.replace('.nii.gz', "_jdet_rda.nii.gz", 1)
vxm.py.utils.save_volfile(jdet[i, 0, :].detach().cpu().numpy().squeeze(), moving_jdet_name)

# save moved image
moved_name = save_fname.replace('.nii.gz', "_warped_rda.nii.gz", 1)
vxm.py.utils.save_volfile(moved_img[i, 0, :].detach().cpu().numpy().squeeze(), moved_name)

# save warp image
warp_rda_name = save_fname.replace('.nii.gz', "_warp_rda.nii.gz", 1)
vxm.py.utils.save_volfile(warp[i, :].permute(1, 2, 3, 0).detach().cpu().numpy().squeeze(), warp_rda_name)
















