#!/usr/bin/env python

"""
Example script to test a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.
"""

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

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import vxm_model as vxm  # nopep8
from vxm_model.py.utils import RunningAverage, get_logger, jacobian_determinant, get_moving_volume, get_moved_volume
from vxm_model.torch import layers

# parse the commandline
parser = argparse.ArgumentParser()

############################# data organization parameters ############################
parser.add_argument('--curr-machine', default='lambda',
                    help='specify on which machine the model is running')
parser.add_argument('--gpu', default='1', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--model-dir', default='Model',
                    help='model output directory (default: Model)')
parser.add_argument('--data-dir', default='final_paper',
                    help='the data split used for this experiment (default: final_paper)')
parser.add_argument('--root-dir', metavar='DIR',
                    default="/Longi_T1_2GO_QC",
                    help='path to dataset')
parser.add_argument('--test-stages', default="[0, 1, 3, 5]",  # "[0, 1, 3, 5]" # '[dump]'
                    help='Input stages to test the model')
parser.add_argument('--ret-affine', default=True,
                    help='Whether to keep affine transformations to save .nii.gz file')
parser.add_argument('--random-warp-direct', default=False,
                    help='Whether to ramdomly input warp direction or not, True for voxelmorph, False for ALOHA')

############################# test parameters ############################
parser.add_argument('--n-classes', type=int, default=2, help='number of output classes for unet segmentation')
                                   # default = 2: [nothing, shrink, expand]
parser.add_argument('--in-dim', type=int, default=2, help='default dimension for unet seg input')
                    # 2: input 2 images for [bl1, bl2] for a single attention map (num_attn_maps=1),
                    #                    or [bl, fu] for two attention maps,each attention map (num_attn_maps=2)
                    # 4: input 4 images for [bl1, fu1, bl2, fu2] for a single attention map(num_attn_maps=1)
parser.add_argument('--out-dim', type=int, default=3, help='default dimension for unet seg output')
                    # 1: output attention map directly,
                    # 2: output 2 channels mimicking segmentation images
                    # 3: output 3 channels of [shrink, bg, expand] voxels
parser.add_argument('--num-attn-maps', type=int, default=2, help='default number of attention maps for two pairs')
                    # 1: a single attention map for two input pairs, required same data augmentation for both pairs
                    # 2: generate separate attention maps for each input pair.
parser.add_argument('--hyper-a-sigmoid', type=float, default=0.02, help='default hyper parameter for sigmoid for RISI loss')
parser.add_argument('--risi-categories', type=int, default=8, help='default number of label categories for RISI loss')
parser.add_argument('--risi-loss-weight', type=float, default=1, help='default weight of risi loss')
parser.add_argument('--test-batch-size', type=int, default=50, help='batch size (default: 1)')
parser.add_argument('--steps-per-test-epoch', type=int, default=500, help='how many steps per test epoch')
parser.add_argument('--log-after-iters', type=int, default=20, help='')

############################# network architecture parameters ############################
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

############################ loss hyperparameters ############################
parser.add_argument('--test-grp', default="test", # train, val, test
                    help=' "val" or "test", first test on evaluation group, last test on test group ')
parser.add_argument('--segs', default=True,
                    help='Whether to include segmentation in data loading,' +
                         'set to True for Jacobian calculation (default: True)')
parser.add_argument('--load-attention',
                    default="/vxm_attention_2_ALOHA_20231030/Model/2023-12-06_13-28/last_checkpoint_0024.pt",
                    # default="/vxm_attention_2_RISI_two_maps_20220727/Model/2022-07-27_21-17/last_checkpoint_0020.pt",
                    help='optional model file to initialize with')
args = parser.parse_args()

if args.curr_machine == "picsl":
    root_dir = "/media/mengjin/MDONG2T"
    vxm_dir = "/home/mengjin/Desktop/Longi_T1_Aim2"

elif args.curr_machine == "lambda":
    root_dir = "/data/mengjin"
    vxm_dir = "/data/mengjin/Longi_T1_Aim2"

args.root_dir = root_dir + args.root_dir


############################## Start Testing #############################
print("Test on", args.load_attention)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.test_batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.test_batch_size, nb_gpus)

print("load model")
model = vxm.networks.VxmSingleAttentionALOHA.load(vxm_dir + args.load_attention, device)

# prepare the model for test and send to device
model.to(device)
model.eval()

# load and prepare test data

test = args.test_grp
test_group = [test]
# test_group = ["test"] / ["val"]
test_stages = args.test_stages.strip('[]').split(', ')
test_img_list = args.root_dir + "/" + test + "_list.csv"
print("Intra-subject registration")
vxm.py.utils.get_file_list_intra_subject(args.root_dir, args.data_dir, test_group, 
                                            test_img_list, test_stages)

test_files = vxm.py.utils.read_file_list(test_img_list,
                                         prefix="",
                                         suffix="")
print("test images file located", test_img_list)
assert len(test_files) > 0, 'Could not find any test data.'
print("Number of test images:", len(test_files))
steps_per_test_epoch = math.ceil(len(test_files)/args.test_batch_size)

# extract shape from sampled input
inshape = (48, 80, 64)

test_augment = ['normalize', 'fixed_crop'] # 'rotate', 
test_generator = vxm.generators.intra_scan_pair(
    test_files,
    test_augment,
    mode="test",
    batch_size=args.test_batch_size,
    bidir=False,
    add_feat_axis=False,
    segs=True,
    output_size=inshape,
    ret_affine=True,
    num_attn_maps=args.num_attn_maps
)

# prepare model directory
model_dir = vxm_dir + args.load_attention[:-3] + test + "_heatmap"
try:
    shutil.rmtree(model_dir)
except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))

os.makedirs(model_dir, exist_ok=True)


# test
epoch_step_time = []

# test
csv_name = model_dir + "/intra-subject_" + test + ".csv"
if os.path.exists(csv_name):
    os.remove(csv_name)
    

with torch.no_grad():
    with open(csv_name, 'a', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(["moving_fname", "fixed_fname",
                        "subjectID", "side", "stage", "bl_time", "fu_time",
                        "date_diff_true", "date_diff_label",
                        "date_diff_pred1", "date_diff_pred2",
                        "date_diff_acc1", "date_diff_acc2",
                        # "vol_diff_ratio_hippo", "vol_diff_ratio_sulcus", 
                        "bl_hippo_volume", "fu_hippo_volume",
                        "bl_sulcus_volume", "fu_sulcus_volume",
                        "hippo_attn_voxel", "sulcus_attn_voxel",])
        
        for step in range(steps_per_test_epoch):
            print("step:", step)

            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            sample= next(test_generator)

            sample["imgs_seg"] = sample["imgs_seg"].to(device).float() # .permute(0, 4, 1, 2, 3)

            sample["imgs_bl"] = sample["imgs_bl"].to(device).float() # .permute(0, 4, 1, 2, 3)
            sample["imgs_fu"] = sample["imgs_fu"].to(device).float() # .permute(0, 4, 1, 2, 3)

            sample["vol_def"] = sample["vol_def"].to(device).float() # .permute(0, 4, 1, 2, 3)

            sample["date_diff_label"] = torch.from_numpy(sample["date_diff_label"]).to(device).long()

            x = torch.cat([sample["imgs_bl"], sample["imgs_fu"] ], dim=1)
            out_seg = model.UNet3D_Seg(x)
            out_seg, moved_img, volume_diff, volume, warped_volume, masks, warped_masks, jdet = model.attention(sample["imgs_bl"],
                            sample["imgs_fu"],
                            sample["vol_def"],
                            out_seg,
                            registration=True)


            volume_diff_divide_hippo, volume_diff_divide_sulcus = volume_diff
            hippo_volume, sulcus_volume = volume
            warped_hippo_volume, warped_sulcus_volume = warped_volume
            hippo_mask, sulcus_mask = masks
            warped_hippo_mask, warped_sulcus_mask = warped_masks

            volume_diff_divide_hippo = 50 * torch.log(volume_diff_divide_hippo)
            volume_diff_divide_hippo = torch.stack((volume_diff_divide_hippo, - volume_diff_divide_hippo), dim=1)

            volume_diff_divide_sulcus = 50 * torch.log(volume_diff_divide_sulcus)
            volume_diff_divide_sulcus = torch.stack((- volume_diff_divide_sulcus, volume_diff_divide_sulcus), dim=1)

            vol_diff_label = [volume_diff_divide_hippo, volume_diff_divide_sulcus]
            attention_pred = torch.argmax(out_seg, dim=1)

            # calculate this number by attention sum of voxels
            hippo_attn_vox = (attention_pred == 1).sum(dim=(1, 2, 3))
            sulcus_attn_vox = (attention_pred == 2).sum(dim=(1, 2, 3))

            date_diff_pred1 = torch.argmax(vol_diff_label[0], dim=1)
            date_diff_pred2 = torch.argmax(vol_diff_label[1], dim=1)
            
            for i in range(len(sample["date_diff_label"])):

                moving_name = sample["img_names"][i].split("/")[-1]
                moving_seg_name = model_dir + "/" + moving_name.replace('blmptrim_', "blmptrim_seg", 1).replace(
                    '_to_hw', '', 1)

                subjectID = moving_name[:10]
                stage = sample["img_names"][i].split("/")[-2]
                side = moving_name.split("_")[-3]
                bl_time = moving_name[11:21]
                fu_time = moving_name[22:32]

                moving_name = model_dir + "/" + moving_name
                fixed_name = moving_name.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)
                moved_name = moving_name.replace('.nii.gz', "_moved.nii.gz", 1)
                warp_name = moving_name.replace('.nii.gz', "_warp.nii.gz", 1)

                # gather information of subjects
                row = [moving_name, fixed_name,
                        subjectID, side, stage, bl_time, fu_time,
                        sample["date_diff_true"][i], sample["date_diff_label"][i].item(),
                        date_diff_pred1[i].item(), date_diff_pred2[i].item(),
                        int(sample["date_diff_label"][i].item() == date_diff_pred1[i].item()), int(sample["date_diff_label"][i].item() == date_diff_pred2[i].item()),
                        #    vol_diff_label[0][i][0].item(), vol_diff_label[1][i][0].item(),
                        hippo_volume[i].item(), warped_hippo_volume[i].item(),
                        sulcus_volume[i].item(), warped_sulcus_volume[i].item(),
                        hippo_attn_vox[i].item(), sulcus_attn_vox[i].item(),
                        ]
                
                print(i)
                # print("vol_diff_label                                                 ", 
                # vol_diff_label[0][0].item(), vol_diff_label[1][0].item())
                print("hippo_volume.item(), warped_hippo_volume.item()                ", 
                      hippo_volume[i].item(), warped_hippo_volume[i].item())
                print("sulcus_volume.item(), warped_sulcus_volume.item()              ", 
                      sulcus_volume[i].item(), warped_sulcus_volume[i].item())
                # print("hippo_attn_vox.item(), sulcus_attn_vox.item()                  ", 
                # hippo_attn_vox.item(), sulcus_attn_vox.item())

                

                wr.writerow(row)
                print(row)

                # save example images, sample the last images of each batch
                affine_trans = [sample["affines"][i, :], sample["orig_shapes"][i, :], sample["crop_posns"][i, :]]
                # save moving image
                # # save fixed image
                # vxm.py.utils.save_volfile(inputs[1][i, 0, :].detach().cpu().numpy().squeeze(), fixed_name, affine_trans)
                shutil.copyfile(sample["img_names"][i], moving_name)  # directly copy original file
                # save attention image
                moving_attention_name = moving_name.replace('.nii.gz', "_attention.nii.gz", 1)
                vxm.py.utils.save_volfile(attention_pred[i, :].detach().cpu().numpy().squeeze().astype(float),
                                            moving_attention_name, affine_trans)
                
                # hippo_mask_name = moving_name.replace('.nii.gz', "hippo_mask.nii.gz", 1)
                # vxm.py.utils.save_volfile(hippo_mask[i, :].detach().cpu().numpy().squeeze().astype(float), hippo_mask_name, affine_trans)

                # sulcus_mask_name = moving_name.replace('.nii.gz', "sulcus_mask.nii.gz", 1)
                # vxm.py.utils.save_volfile(sulcus_mask[i, :].detach().cpu().numpy().squeeze().astype(float), sulcus_mask_name, affine_trans)

                moving_pos_mask = moving_name.replace('.nii.gz', "_position_mask.nii.gz", 1)
                vxm.py.utils.save_volfile(np.ones(inshape), moving_pos_mask, affine_trans)
                # save jacobian determinant
                # moving_jdet_name = moving_name.replace('.nii.gz', "_jdet.nii.gz", 1)
                # vxm.py.utils.save_volfile(jdet[i, 0, :].detach().cpu().numpy().squeeze(),
                #                             moving_jdet_name, affine_trans)

                # # save seg image before registration
                # vxm.py.utils.save_volfile(sample["imgs_seg"][i].detach().cpu().numpy().squeeze(), moving_seg_name, affine_trans)

                # save moved image
                vxm.py.utils.save_volfile(moved_img[i, 0, :].detach().cpu().numpy().squeeze(),
                                            moved_name, affine_trans)

                # get compute time
                epoch_step_time.append(time.time() - step_start_time)
