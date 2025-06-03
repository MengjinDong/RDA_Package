import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use('TkAgg') # MUST BE CALLED BEFORE IMPORTING plt, or qt4agg

from .utils import jacobian_determinant_batch, get_moving_volume, get_moved_volume, my_create_pytorch_grid
from . import layers
from . import losses
from .modelio import LoadableModel, store_config_args

# def sigmoid(x, a):
#     return 1 / (1 + torch.exp(-a * x))

Sigmoid = nn.Sigmoid()

win = [2] * 3
NCC_metric = losses.NCC(win).loss # radius = 2 or window size = 5


def act_4_0(x, y, a):
    sig = torch.max(torch.stack([Sigmoid((2 * x - y) * a) * Sigmoid((-2 * x - y) * a),
                                    Sigmoid((2 * x + y) * a) * Sigmoid((-2 * x + y) * a)], dim=0), dim=0).values
    return sig

def act_4_1(x, y, a):
    sig = torch.max(
        torch.stack([Sigmoid((2 * x - y) * a) * Sigmoid((-x + y) * a),
                        Sigmoid((-2 * x - y) * a) * Sigmoid((x + y) * a),
                        Sigmoid((2 * x + y) * a) * Sigmoid((-x - y) * a),
                        Sigmoid((-2 * x + y) * a) * Sigmoid((x - y) * a)], dim=0), dim=0).values
    return sig

def act_4_2(x, y, a):
    sig = torch.max(
        torch.stack([Sigmoid((0.5 * x - y) * a) * Sigmoid((-x + y) * a),
                        Sigmoid((-0.5 * x - y) * a) * Sigmoid((x + y) * a),
                        Sigmoid((0.5 * x + y) * a) * Sigmoid((-x - y) * a),
                        Sigmoid((-0.5 * x + y) * a) * Sigmoid((x - y) * a)], dim=0), dim=0).values
    return sig

def act_4_3(x, y, a):
    sig = torch.max(torch.stack([Sigmoid((2 * y - x) * a) * Sigmoid((-2 * y - x) * a),
                                    Sigmoid((2 * y + x) * a) * Sigmoid((-2 * y + x) * a)], dim=0), dim=0).values

    return sig

def act_8_0(x, y, a):
    return torch.max(torch.stack([Sigmoid((4 * x - y) * a) * Sigmoid((-4 * x - y) * a),
                                    Sigmoid((4 * x + y) * a) * Sigmoid((-4 * x + y) * a)], dim=0), dim=0).values

def act_8_1(x, y, a):
    return torch.max(
        torch.stack([Sigmoid((4 * x - y) * a) * Sigmoid((-2 * x + y) * a),
                        Sigmoid((-4 * x - y) * a) * Sigmoid((2 * x + y) * a),
                        Sigmoid((4 * x + y) * a) * Sigmoid((-2 * x - y) * a),
                        Sigmoid((-4 * x + y) * a) * Sigmoid((2 * x - y) * a)], dim=0), dim=0).values

def act_8_2(x, y, a):
    return torch.max(
        torch.stack([Sigmoid((2 * x - y) * a) * Sigmoid((-4 * x + 3 * y) * a),
                        Sigmoid((-2 * x - y) * a) * Sigmoid((4 * x + 3 * y) * a),
                        Sigmoid((2 * x + y) * a) * Sigmoid((-4 * x - 3 * y) * a),
                        Sigmoid((-2 * x + y) * a) * Sigmoid((4 * x - 3 * y) * a)], dim=0), dim=0).values

def act_8_3(x, y, a):
    return torch.max(
        torch.stack([Sigmoid((4 * x - 3 * y) * a) * Sigmoid((-x + y) * a),
                        Sigmoid((-4 * x - 3 * y) * a) * Sigmoid((x + y) * a),
                        Sigmoid((4 * x + 3 * y) * a) * Sigmoid((-x - y) * a),
                        Sigmoid((-4 * x + 3 * y) * a) * Sigmoid((x - y) * a)], dim=0), dim=0).values

def act_8_4(x, y, a):
    return torch.max(
        torch.stack([Sigmoid((3 * x - 4 * y) * a) * Sigmoid((-x + y) * a),
                        Sigmoid((-3 * x - 4 * y) * a) * Sigmoid((x + y) * a),
                        Sigmoid((3 * x + 4 * y) * a) * Sigmoid((-x - y) * a),
                        Sigmoid((-3 * x + 4 * y) * a) * Sigmoid((x - y) * a)], dim=0), dim=0).values

def act_8_5(x, y, a):
    return torch.max(
        torch.stack([Sigmoid((x - 2 * y) * a) * Sigmoid((-3 * x + 4 * y) * a),
                        Sigmoid((-x - 2 * y) * a) * Sigmoid((3 * x + 4 * y) * a),
                        Sigmoid((x + 2 * y) * a) * Sigmoid((-3 * x - 4 * y) * a),
                        Sigmoid((-x + 2 * y) * a) * Sigmoid((3 * x - 4 * y) * a)], dim=0), dim=0).values

def act_8_6(x, y, a):
    return torch.max(
        torch.stack([Sigmoid((x - 4 * y) * a) * Sigmoid((-x + 2 * y) * a),
                        Sigmoid((-x - 4 * y) * a) * Sigmoid((x + 2 * y) * a),
                        Sigmoid((x + 4 * y) * a) * Sigmoid((-x - 2 * y) * a),
                        Sigmoid((-x + 4 * y) * a) * Sigmoid((x - 2 * y) * a)], dim=0), dim=0).values

def act_8_7(x, y, a):
    return torch.max(torch.stack([Sigmoid((4 * y - x) * a) * Sigmoid((-4 * y - x) * a),
                                    Sigmoid((4 * y + x) * a) * Sigmoid((-4 * y + x) * a)], dim=0), dim=0).values


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class UNet3D_Seg(LoadableModel):
    @store_config_args
    def __init__(self, in_dim = 2, n_classes = 1, num_filters = 8, out_dim = 2):
        super(UNet3D_Seg, self).__init__()

        self.in_dim = in_dim
        self.n_classes = n_classes
        self.num_filters = num_filters
        self.out_dim = out_dim + 1
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = self.conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = self.max_pooling_3d()
        self.down_2 = self.conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = self.max_pooling_3d()
        self.down_3 = self.conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = self.max_pooling_3d()
        self.down_4 = self.conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = self.max_pooling_3d()
        # self.down_5 = self.conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        # self.pool_5 = self.max_pooling_3d()

        # Bridge
        self.bridge = self.conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)

        # Up sampling
        # self.trans_1 = self.conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        # self.up_1 = self.conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_1 = self.conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_1 = self.conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_2 = self.conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_2 = self.conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_3 = self.conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_3 = self.conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_4 = self.conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_4 = self.conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out_seg = self.conv_block_3d(self.num_filters, self.num_filters, activation)
        self.out_seg_final =  nn.Conv3d(self.num_filters, n_classes + 1, kernel_size=1, stride=1, padding=0)

        self.out_jac = self.conv_block_3d(self.num_filters, self.num_filters, activation)
        self.out_jac_final = nn.Conv3d(self.num_filters, 1, kernel_size=1, stride=1, padding=0)

        self.out_synth = self.conv_block_3d(self.num_filters, self.num_filters, activation)
        self.out_synth_final = nn.Conv3d(self.num_filters, self.out_dim, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim = 1)

        self.fc = nn.Linear(245768, self.n_classes)

    def forward(self, x1):
        # Down sampling
        down_1 = self.down_1(x1)  # -> [1, 4, 48, 80, 64]
        pool_1 = self.pool_1(down_1)  # -> [1, 4, 24, 40, 32]

        down_2 = self.down_2(pool_1)  # -> [1, 8, 24, 40, 32]
        pool_2 = self.pool_2(down_2)  # -> [1, 8, 12, 20, 16]

        down_3 = self.down_3(pool_2)  # -> [1, 16, 12, 20, 16]
        pool_3 = self.pool_3(down_3)  # -> [1, 16, 6, 10, 8]

        down_4 = self.down_4(pool_3)  # -> [1, 32, 6, 10, 8]
        pool_4 = self.pool_4(down_4)  # -> [1, 32, 3, 5, 4]

        # down_5 = self.down_5(pool_4)  # -> [1, 64, 3, 5, 4]
        # pool_5 = self.pool_5(down_5)  # -> [1, 64, 1, 2, 2]

        # Bridge
        bridge = self.bridge(pool_4)  # -> [1, 128, 3, 5, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [1, 128, 6, 10, 8] # transcomposition increase the size by 2
        concat_1 = torch.cat([trans_1, down_4], dim=1)  # -> [1, 192, 6, 10, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64,6, 10, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 12, 20, 16]
        concat_2 = torch.cat([trans_2, down_3], dim=1)  # -> [1, 96, 12, 20, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 12, 20, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 24, 40, 32]
        concat_3 = torch.cat([trans_3, down_2], dim=1)  # -> [1, 48, 24, 40, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 24, 40, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 48, 80, 64]
        concat_4 = torch.cat([trans_4, down_1], dim=1)  # -> [1, 24, 48, 80, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 48, 80, 64]

        # trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        # concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> [1, 12, 128, 128, 128]
        # up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        out_seg = self.out_seg(up_4)
        out_seg = self.out_seg_final(out_seg)

        # out_synth = self.out_synth(up_4)  # -> [1, 4, 128, 128, 128] # 2 classes
        # out_synth = self.out_synth_final(out_synth) # -> [1, 2, 128, 128, 128] # 2 classes

        # out_jac_exp = out_jac * 0.1
        # out_jac_exp = torch.exp(out_jac_exp)

        return out_seg


    def conv_block_3d(self, in_dim, out_dim, activation):
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            activation,)

    def conv_trans_block_3d(self, in_dim, out_dim, activation):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            activation,)


    # def conv_trans_block_3d(self, in_dim, out_dim, activation):
    #     return nn.Sequential(
    #         nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.BatchNorm3d(out_dim),
    #         activation,)


    def max_pooling_3d(self):
        return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


    def conv_block_2_3d(self, in_dim, out_dim, activation):
        return nn.Sequential(
            self.conv_block_3d(in_dim, out_dim, activation),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),)


class SingleAttentionALOHA(LoadableModel):
    @store_config_args
    def __init__(self, inshape, 
                 unet_model = UNet3D_Seg(),
                 num_reg_labels=4,
                 hyper_a=0.5,
                 risi_categories = 4):
        super(SingleAttentionALOHA, self).__init__()

        self.UNet3D_Seg = unet_model
        self.num_reg_labels = num_reg_labels
        self.hyper_a = hyper_a
        self.risi_categories = risi_categories
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.transformer = layers.SpatialTransformer(inshape)


    def act_4_categories(self, x, y, a):
        return torch.stack([act_4_0(x, y, a),
                            act_4_1(x, y, a),
                            act_4_2(x, y, a),
                            act_4_3(x, y, a)], dim=1)

    def act_8_categories(self, x, y, a):
        return torch.stack([act_8_0(x, y, a),
                            act_8_1(x, y, a),
                            act_8_2(x, y, a),
                            act_8_3(x, y, a),
                            act_8_4(x, y, a),
                            act_8_5(x, y, a),
                            act_8_6(x, y, a),
                            act_8_7(x, y, a)], dim=1)

    def attention(self, img1, img2, warp, out_seg, registration=True):

        if torch.sum(torch.isnan(img1)):
            print("nan found in source img1 image!")

        if torch.sum(torch.isnan(img2)):
            print("nan found in target img2 image!")

        # ALOHA branch
        jdet = jacobian_determinant_batch(warp)

        if torch.sum(torch.isnan(out_seg)):
            print("nan found in out_seg!")

        out_seg = self.softmax(100 * out_seg) # scale was 100, now 0
        hippo_mask = out_seg[:, 1:2, :, :, :]  # 0 bg, 1 hippo
        sulcus_mask = out_seg[:, 2:3, :, :, :]  # 0 bg, 2 sulcus

        # attention_pred = torch.argmax(out_seg, dim=1)

        # # thresholding at 0.5 is different from argmax
        # # hippo_mask_binary = (hippo_mask > 0.5).float()
        # # sulcus_mask_binary = (sulcus_mask > 0.5).float()

        # # now take argmax to get binary image
        # hippo_mask_binary = (attention_pred == 1).float().unsqueeze(dim=1)
        # sulcus_mask_binary = (attention_pred == 2).float().unsqueeze(dim=1)

        # vxm_moved_img, VXM_warp = self.VxmDense(img1, img2, registration=registration)

        # img1_n = img1.permute(0, 4, 1, 2, 3)
        # img2_n = img2.permute(0, 4, 1, 2, 3)
        # vxm_moved_img_n = vxm_moved_img.permute(0, 4, 1, 2, 3)

        # NCC_img1_img2 = NCC_metric(img1, img2)
        # NCC_img1_warp = NCC_metric(img1, vxm_moved_img)
        # NCC_warp_img2 = NCC_metric(vxm_moved_img, img2)
        # print("VXM NCC_before_reg, NCC_after_reg, NCC_for_reg = ", NCC_img1_img2, NCC_warp_img2, NCC_img1_warp)

        # for hippocampus and sulcus: directly warp the mask

        # transformation using Paul's code
        grid = my_create_pytorch_grid(img1.permute(1, 0, 2, 3, 4).shape).to('cuda')
        grid = grid.repeat(img1.shape[0], 1, 1, 1, 1)
        warp = warp.permute(0, 2, 3, 4, 1)

        # warp img_1n to img_2n
        moved_img = F.grid_sample(img2, grid + warp, align_corners=False)

        # NCC_img1_img2 = NCC_metric(img1, img2)
        # NCC_img1_warp = NCC_metric(img1, moved_img)
        # NCC_warp_img2 = NCC_metric(moved_img, img2)
        # print("NCC_before_reg, NCC_after_reg, NCC_for_reg = ", NCC_img1_img2, NCC_img1_warp, NCC_warp_img2)

        ## warp the follow-up image to the baseline image
        warped_hippo_mask = F.grid_sample(hippo_mask, grid + warp, align_corners=False)
        
        # in this way, hippo_volume is always the volume of img1
        # warped_hippo_volume is always the volume of img2

        warped_hippo_volume = get_moving_volume(hippo_mask) 
        hippo_volume = get_moving_volume(warped_hippo_mask)

        # sulcus
        # warped_sulcus_mask = self.transformer(sulcus_mask, warp)
        warped_sulcus_mask = F.grid_sample(sulcus_mask, grid + warp, align_corners=False)

        warped_sulcus_volume = get_moving_volume(sulcus_mask)
        sulcus_volume = get_moving_volume(warped_sulcus_mask)

        # calculate NCC between img1, img2, and moved_img
        # moved_img should look similar to img2
        # TODO: check volumes are in the correct direction after aloha

        volume_diff_divide_hippo = warped_hippo_volume / (hippo_volume + 1e-7)

        volume_diff_divide_sulcus = warped_sulcus_volume / (sulcus_volume + 1e-7)

        if torch.sum(torch.isnan(volume_diff_divide_hippo)):
            print("nan found in hippo!")

        if torch.sum(torch.isnan(volume_diff_divide_sulcus)):
            print("nan found in sulcus!")

        return [out_seg, moved_img,
                [volume_diff_divide_hippo,   volume_diff_divide_sulcus],
                [hippo_volume,        sulcus_volume],
                [warped_hippo_volume, warped_sulcus_volume],
                # [hippo_mask,        sulcus_mask],
                # [warped_hippo_mask, warped_sulcus_mask],               
                jdet
               ]


    def forward(self, bl1, fu1, bl2, fu2, warp1, warp2, registration=True):
        # print("bl1.shape", bl1.shape)
        # print("warp1.shape", warp1.shape)
        # attention branch

        x1 = torch.cat([bl1, fu1], dim=1)
        out_seg1 = self.UNet3D_Seg(x1)
        x2 = torch.cat([bl2, fu2], dim=1)
        out_seg2 = self.UNet3D_Seg(x2)

        # no switching order for image pair input, warp the follow-up image to the baseline image
        # moved_img1 should look closest to bl1.

        out_seg1, moved_img1, volume_diff1, volume1, warped_volume1, jdet1 = self.attention(bl1, 
                                                                                            fu1, warp1,
                                                                                            out_seg1,
                                                                                            registration=registration)

        out_seg2, moved_img2, volume_diff2, volume2, warped_volume2, jdet2 = self.attention(bl2, fu2, warp2,
                                                                                            out_seg2,
                                                                                            registration=registration)

        volume_diff_divide_hippo1, volume_diff_divide_sulcus1 = volume_diff1
        volume_diff_divide_hippo2, volume_diff_divide_sulcus2 = volume_diff2

        hippo_volume1, sulcus_volume1 = volume1
        hippo_volume2, sulcus_volume2 = volume2

        warped_hippo_volume1, warped_sulcus_volume1 = warped_volume1
        warped_hippo_volume2, warped_sulcus_volume2 = warped_volume2

        volume_diff_subtract_hippo1 = (warped_hippo_volume1 - hippo_volume1)
        volume_diff_subtract_hippo2 = (warped_hippo_volume2 - hippo_volume2)

        volume_diff_subtract_sulcus1 = (warped_sulcus_volume1 - sulcus_volume1)
        volume_diff_subtract_sulcus2 = (warped_sulcus_volume2 - sulcus_volume2)

        # method 3: To calculate volume_diff_ratio_sulcus and volume_diff_ratio_hippo
        # Use volume_diff_hippo1 and volume_diff_sulcus1 to calculate

        # print(volume_diff_subtract_hippo1.detach().cpu().numpy(), volume_diff_subtract_hippo2.detach().cpu().numpy(),
        #       volume_diff_subtract_sulcus1.detach().cpu().numpy(), volume_diff_subtract_sulcus2.detach().cpu().numpy())
        if  self.risi_categories == 4:
            volume_diff_ratio_hippo = self.act_4_categories(volume_diff_subtract_hippo1, volume_diff_subtract_hippo2, self.hyper_a)
            volume_diff_ratio_sulcus = self.act_4_categories(volume_diff_subtract_sulcus1, volume_diff_subtract_sulcus2, self.hyper_a)
        elif self.risi_categories == 8:
            volume_diff_ratio_hippo = self.act_8_categories(volume_diff_subtract_hippo1, volume_diff_subtract_hippo2, self.hyper_a)
            volume_diff_ratio_sulcus = self.act_8_categories(volume_diff_subtract_sulcus1, volume_diff_subtract_sulcus2, self.hyper_a)


        # To calculate volume change score for STO loss
        # mimic the atrophy measurement, calculating average Jacobian
        volume_diff_divide_hippo1 = 50 * torch.log(volume_diff_divide_hippo1)
        volume_diff_divide_hippo1 = torch.stack((volume_diff_divide_hippo1, - volume_diff_divide_hippo1), dim=1)

        volume_diff_divide_sulcus1 = 50 * torch.log(volume_diff_divide_sulcus1)
        volume_diff_divide_sulcus1 = torch.stack((- volume_diff_divide_sulcus1, volume_diff_divide_sulcus1), dim=1)

        # mimic the atrophy measurement, calculating average Jacobian
        volume_diff_divide_hippo2 = 50 * torch.log(volume_diff_divide_hippo2)
        volume_diff_divide_hippo2 = torch.stack((volume_diff_divide_hippo2, - volume_diff_divide_hippo2), dim=1)

        volume_diff_divide_sulcus2 = 50 * torch.log(volume_diff_divide_sulcus2)
        volume_diff_divide_sulcus2 = torch.stack((- volume_diff_divide_sulcus2, volume_diff_divide_sulcus2), dim=1)
        # return attention maps for both pairs
        # return STO loss for the first pair
        # return STO loss for the second pair
        # return RISI loss

        return out_seg1, out_seg2, \
                moved_img1, moved_img2, \
                [volume_diff_divide_hippo1, volume_diff_divide_sulcus1, \
                volume_diff_divide_hippo2, volume_diff_divide_sulcus2, \
                volume_diff_ratio_hippo, volume_diff_ratio_sulcus]

