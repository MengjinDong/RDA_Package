import os
import sys
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from . import utils
import pandas as pd
import pandas as pd


class DatasetRISI3D(Dataset):
    """
    Dataset for RISI 3D data.
    """

    def __init__(self, csv_list, 
                 augment,  
                 add_feat_axis=True,
                 output_size=(96, 160, 128),
                 risi_categories=4,
                 ret_affine=False,
                 max_angle=45,
                 rotate_prob=0.5):

        self.csv_list = csv_list
        self.augment = augment
        self.add_feat_axis = add_feat_axis
        self.ret_affine = ret_affine
        self.output_size = output_size
        self.risi_categories = risi_categories
        self.max_angle = max_angle
        self.rotate_prob = rotate_prob

        self.image_frame = pd.read_csv(csv_list)

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):       

        sample = {} 

        random_bl1 = self.image_frame.iloc[idx]["bl_fname1"]
        random_bl2 = self.image_frame.iloc[idx]["bl_fname2"]

        sample["bl_fname1"] = random_bl1
        sample["bl_fname2"] = random_bl2

        sample["bl_time1"] = self.image_frame.iloc[idx]["bl_time1"]
        sample["bl_time2"]  = self.image_frame.iloc[idx]["bl_time2"]
        sample["fu_time1"] = self.image_frame.iloc[idx]["fu_time1"]
        sample["fu_time2"] = self.image_frame.iloc[idx]["fu_time2"]

        sample["stage"] = self.image_frame.iloc[idx]["stage"]

        sample["date_diff1"] = float(self.image_frame.iloc[idx]["date_diff1"])
        sample["date_diff2"] = float(self.image_frame.iloc[idx]["date_diff2"])

        label_date_diff1 = float(self.image_frame.iloc[idx]["label_date_diff1"])
        label_date_diff2 = float(self.image_frame.iloc[idx]["label_date_diff2"])
        label_time_interval = float(self.image_frame.iloc[idx]["label_time_interval"])

        sample['label_date_diff1'] = torch.from_numpy(np.array(label_date_diff1).copy()).float()
        sample['label_date_diff2'] = torch.from_numpy(np.array(label_date_diff2).copy()).float()
        sample['label_time_interval'] = torch.from_numpy(np.array(label_time_interval).copy()).float()

        sample['subjectID'] = self.image_frame.iloc[idx]["subjectID"]
        # side = self.image_frame.iloc[idx]["side"]
        sample['side'] = self.image_frame.iloc[idx]["side"] if "side" in self.image_frame.columns else ""

        random_bl1 = ''.join(random_bl1)
        random_fu1 = self.image_frame.iloc[idx]["fu_fname1"]
        random_warp1 = self.image_frame.iloc[idx]["warp_fname1"]
        random_mask1 = self.image_frame.iloc[idx]["seg_fname1"]

        random_bl2 = ''.join(random_bl2)
        random_fu2 = self.image_frame.iloc[idx]["fu_fname2"]
        random_warp2 = self.image_frame.iloc[idx]["warp_fname2"]
        random_mask2 = self.image_frame.iloc[idx]["seg_fname2"]

        ########### load images
        bl_cube1, affine_bl1 = utils.my_read_pytorch_image_from_nifti(random_bl1, dtype=torch.float32, device='cuda')
        fu_cube1, affine_fu1 = utils.my_read_pytorch_image_from_nifti(random_fu1, dtype=torch.float32, device='cuda')
        if self.ret_affine:
            if not np.array_equal(affine_bl1, affine_fu1):
                print(random_bl1, "bl1 and fu1 are not in the same space!")
        warp_orig1, affine_def1 = utils.my_read_pytorch_warp_from_nifti(random_warp1, dtype=torch.float32, device='cuda')

        # print datatype and content of ramdom_mask1
        if os.path.exists(str(random_mask1)):
            mask_cube1, affine_seg1 = utils.my_read_pytorch_image_from_nifti(random_mask1, dtype=torch.float32, device='cuda')
        else:
            mask_cube1 = (bl_cube1 > 1).to(torch.float32)
            affine_seg1 = affine_bl1

        # mask out the blank regions in the warp
        # warp_cube1 shape:  torch.Size([1, 128, 94, 98, 3])
        # mask_cube1 shape:  torch.Size([1, 1, 128, 94, 98])  

        warp_cube1 = (warp_orig1.permute(0, 4, 1, 2, 3) * mask_cube1.expand(-1, 3, -1, -1, -1)).permute(0, 2, 3, 4, 1)

        sample["affines1"] = affine_bl1
        sample["orig_shapes1"] = bl_cube1.shape

        # if np.array_equal(affine_bl1, affine_fu1) and np.array_equal(affine_fu1, affine_seg1):
        #     sample["affines1"] = affine_bl1
        #     sample["orig_shapes1"] = bl_cube1.shape
        # else:
        #     print("\n", random_bl2, "affine deformations are not in the same space!")
        #     print("affine_bl1: ", affine_bl1)
        #     print("affine_fu1: ", affine_fu1)
        #     print("affine_def1: ", affine_def1)
        #     print("affine_seg1: ", affine_seg1)

        bl_cube2, affine_bl2 = utils.my_read_pytorch_image_from_nifti(random_bl2, dtype=torch.float32, device='cuda')
        fu_cube2, affine_fu2 = utils.my_read_pytorch_image_from_nifti(random_fu2, dtype=torch.float32, device='cuda')
        warp_orig2, affine_def2 = utils.my_read_pytorch_warp_from_nifti(random_warp2, dtype=torch.float32, device='cuda')
        if os.path.exists(str(random_mask2)):
            mask_cube2, affine_seg2 = utils.my_read_pytorch_image_from_nifti(random_mask2, dtype=torch.float32, device='cuda')
        else:
            mask_cube2 = (bl_cube2 > 1).to(torch.float32)
            affine_seg2 = affine_bl2

        # mask out the blank regions in the warp
        warp_cube2 = (warp_orig2.permute(0, 4, 1, 2, 3) * mask_cube2.expand(-1, 3, -1, -1, -1)).permute(0, 2, 3, 4, 1)

        sample["affines2"] = affine_bl2
        sample["orig_shapes2"] = bl_cube2.shape

        # if np.array_equal(affine_bl2, affine_fu2) and np.array_equal(affine_fu2, affine_seg2):
        #     sample["affines2"] = affine_bl2
        #     sample["orig_shapes2"] = bl_cube2.shape
        # else:
        #     print("\n", random_bl2, "affine deformations are not in the same space!")

        # print(random_bl1, "aug1")

        ########### augmentation
        sample["imgs_bl1"], \
        sample["imgs_fu1"], \
        sample["crop_posns1"], \
        sample["imgs_seg1"], \
        sample["imgs_warp1"] = utils.single_sample_aug(bl_cube1,
                                                        fu_cube1,
                                                        warp_cube1,
                                                        mask_cube1,
                                                        augment=self.augment,
                                                        add_feat_axis=self.add_feat_axis,
                                                        max_angle=self.max_angle,
                                                        rotate_prob=self.rotate_prob,
                                                        # output_size=[48, 80, 64],
                                                        output_size=self.output_size)

        # print(random_bl2, "aug2")

        sample["imgs_bl2"], \
        sample["imgs_fu2"], \
        sample["crop_posns2"], \
        sample["imgs_seg2"], \
        sample["imgs_warp2"]  = utils.single_sample_aug(bl_cube2,
                                                            fu_cube2,
                                                            warp_cube2,
                                                            mask_cube2,
                                                            augment=self.augment,
                                                            add_feat_axis=self.add_feat_axis,
                                                            max_angle=self.max_angle,
                                                            rotate_prob=self.rotate_prob,
                                                            # output_size=[48, 80, 64],
                                                            output_size=self.output_size)

        return sample


class DatasetRISI3DPair(Dataset):
    """
    Dataset for RISI 3D data for only one image pair.
    """

    def __init__(self, csv_list, 
                 augment,  
                 add_feat_axis=True,
                 output_size=(96, 160, 128),
                 ret_affine=False,
                 max_angle=0,
                 rotate_prob=0):

        self.csv_list = csv_list
        self.augment = augment
        self.add_feat_axis = add_feat_axis
        self.ret_affine = ret_affine
        self.output_size = output_size
        self.max_angle = max_angle
        self.rotate_prob = rotate_prob

        self.image_frame = pd.read_csv(csv_list)

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):       

        sample = {} 

        random_bl = self.image_frame.iloc[idx]["bl_fname"]

        sample["bl_fname"] = random_bl
        sample["bl_time"] = self.image_frame.iloc[idx]["bl_time"]
        sample["fu_time"] = self.image_frame.iloc[idx]["fu_time"]

        sample["stage"] = self.image_frame.iloc[idx]["stage"]
        sample["date_diff"] = float(self.image_frame.iloc[idx]["date_diff"])
        
        label_date_diff = float(self.image_frame.iloc[idx]["label_date_diff"])
        sample['label_date_diff'] = torch.from_numpy(np.array(label_date_diff).copy()).float()
        sample['subjectID'] = self.image_frame.iloc[idx]["subjectID"]
        # side = self.image_frame.iloc[idx]["side"]
        sample['side'] = self.image_frame.iloc[idx]["side"] if "side" in self.image_frame.columns else ""

        random_bl = ''.join(random_bl)
        random_fu = self.image_frame.iloc[idx]["fu_fname"]
        random_warp = self.image_frame.iloc[idx]["warp_fname"]
        random_mask = self.image_frame.iloc[idx]["seg_fname"]

        ########### load images
        bl_cube, affine_bl = utils.my_read_pytorch_image_from_nifti(random_bl, dtype=torch.float32, device='cuda')
        fu_cube, affine_fu = utils.my_read_pytorch_image_from_nifti(random_fu, dtype=torch.float32, device='cuda')
        if self.ret_affine:
            if not np.array_equal(affine_bl, affine_fu):
                print(random_bl, "bl and fu are not in the same space!")
        warp_cube, affine_def = utils.my_read_pytorch_warp_from_nifti(random_warp, dtype=torch.float32, device='cuda')

        # print datatype and content of ramdom_mask
        if os.path.exists(str(random_mask)):
            mask_cube, affine_seg = utils.my_read_pytorch_image_from_nifti(random_mask, dtype=torch.float32, device='cuda')
        else:
            mask_cube = (bl_cube > 1).to(torch.float32)
            affine_seg = affine_bl

        sample["affines"] = affine_bl
        sample["orig_shapes"] = bl_cube.shape

        # if np.array_equal(affine_bl, affine_fu) and np.array_equal(affine_fu, affine_seg):
        #     sample["affines"] = affine_bl
        #     sample["orig_shapes"] = bl_cube.shape
        # else:
        #     print("\n", random_bl, "affine deformations are not in the same space!")
        #     print("affine_bl: ", affine_bl)
        #     print("affine_fu: ", affine_fu)
        #     print("affine_def: ", affine_def)
        #     print("affine_seg: ", affine_seg)

        ########### augmentation
        sample["imgs_bl"], \
        sample["imgs_fu"], \
        sample["crop_posns"], \
        sample["imgs_seg"], \
        sample["imgs_warp"] = utils.single_sample_aug(bl_cube,
                                                        fu_cube,
                                                        warp_cube,
                                                        mask_cube,
                                                        augment=self.augment,
                                                        add_feat_axis=self.add_feat_axis,
                                                        max_angle=self.max_angle,
                                                        rotate_prob=self.rotate_prob,
                                                        # output_size=[48, 80, 64],
                                                        output_size=self.output_size)

        return sample
