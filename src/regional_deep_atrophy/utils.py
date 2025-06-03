# internal python imports
import csv
import torch
from tensorboardX import SummaryWriter
import sys
from itertools import permutations, combinations
import torch.nn.functional as F
import time

# third party imports
import numpy as np
import glob
import math
from datetime import datetime
# local/our imports
import pystrum.pynd.ndutils as nd
from . import data_aug_gpu
import logging
import matplotlib.pyplot as plt
import nibabel as nib
import shutil
import random

date_format = "%Y-%m-%d"


def my_pytorch_coord_transform(img_size):
    S = np.eye(3)[(2,1,0),:]
    W = S @ (np.diag(2.0 / np.array(img_size)))
    z = S @ (1.0 / np.array(img_size) - 1)
    return W,z

def my_apply_affine_to_pytorch_grid(A, b, grid):
    px = grid[0,:,:,:,0]
    py = grid[0,:,:,:,1]
    pz = grid[0,:,:,:,2]    
    qx = A[0,0]*px + A[0,1]*py + A[0,2]*pz + b[0]
    qy = A[1,0]*px + A[1,1]*py + A[1,2]*pz + b[1]
    qz = A[2,0]*px + A[2,1]*py + A[2,2]*pz + b[2]
    return torch.stack((qx,qy,qz),3).unsqueeze(0).type(grid.dtype)

def my_create_pytorch_grid(img_tensor_dim, **kwargs):
    """
    Generate an identity grid for use with grid_sample, similar to meshgrid in NumPy
    
    :param img_tensor_dim: Dimensions of tensor holding the reference image (tuple of 5 values)
    :param dtype: PyTorch data type for the grid (see torch.Tensor)
    :param device: PyTorch device for the grid (see torch.Tensor)
    :returns: 5D tensor of dimension [1,S_x,S_y,S_z,3] containing the grid
    """
    
    # Generate a 3x4 representation of and identity matrix
    T_idmat = torch.eye(3,4).unsqueeze(0)

    # Update the type and device of the grid
    T_dummy = torch.zeros(1, **kwargs)
        
    # Generate a sampling grid inside of a no_grad block
    T_grid = F.affine_grid(T_idmat, img_tensor_dim, align_corners=False).type(T_dummy.dtype).to(T_dummy.device)
    
    return T_grid

def my_read_pytorch_image_from_nifti(filename, **kwargs):
    """
    Read a 3D or 4D scalar image from a NIfTI file, return as a PyTorch tensor
    
    :param filename:    Path to a NIfTI file storing the image
    
    :Keyword Arguments: You can supply parameters 'dtype' and 'device' that will be used to
                        initialize the PyTorch tensor
                        
    :returns:           Shape [1,C,D,H,W] PyTorch tensor holding the image
    """
    img = nib.load(filename)
    T = torch.tensor(img.get_fdata(), **kwargs)
    while len(T.shape) < 5:
        T = T.unsqueeze(0)  
    return T, img.affine

def my_read_pytorch_warp_from_nifti(filename, **kwargs):
    """
    Read an ANTs/Greedy displacement field from a NIfTI file and return as a PyTorch tensor.
    
    :param filename:    Path to a NIfTI file storing the displacement
    
    :Keyword Arguments: You can supply parameters 'dtype' and 'device' that will be used to
                        initialize the PyTorch tensor
                        
    :returns:           Shape [1,D,H,W,3] PyTorch tensor holding the displacement field. The
                        displacement field will be in the PyTorch normalized coordinates, i.e.,
                        see torch.nn.functional.grid_sample
    """
    
    # Read the data
    img = nib.load(filename)
    
    # Extract the data, convert to PyTorch, and permute axes to match affine_grid
    T = torch.tensor(img.get_fdata(), **kwargs).permute(3,0,1,2,4)
    
    # Displacements in Greedy are in physical units. They need to be converted to voxel units
    # and then to the PyTorch normalized coordinate system. Mapping to voxel units is given
    # by the inverse of the NIfTI s-form matrix. Mapping to PyTorch units is from the first 
    # homework asssignment. Since we are transforming displacement vectors and not points, we
    # do not need to worry about the translation component of the affine transform
    W, _ = my_pytorch_coord_transform(img.shape[:3])
    S_inv = np.linalg.inv(img.affine[:3,:3])
    M_LPS = np.diag(np.array([-1,-1,1]))
    A = W @ S_inv @ M_LPS
    
    # Apply the transformation A to the displacements
    T_pt = my_apply_affine_to_pytorch_grid(A, torch.zeros(3), T) 
    
    return T_pt, img.affine


def single_sample_aug(vol_bl,
                      vol_fu,
                      vol_def,
                      vol_seg=None,
                      augment='normalize',
                      add_feat_axis=False,
                      max_angle=15,
                      rotate_prob=0.5,
                      output_size=(48, 80, 64)):

    if 'normalize' in augment:
        vol_bl, vol_fu = data_aug_gpu.Normalize([vol_bl, vol_fu])

    if 'erase' in augment:
        vol_bl, vol_fu = data_aug_gpu.randomErase3d([vol_bl, vol_fu])

    segs = vol_seg is not None
    if segs:
        image_list1 = [vol_bl, vol_fu, vol_seg, vol_def]
    else:
        image_list1 = [vol_bl, vol_fu, vol_def]

    # image_list2 = copy.deepcopy(image_list1)

    if 'rotate' in augment:
        rotate_prob = 1
        image_list1 = data_aug_gpu.randomRotation3d(image_list1, max_angle, rotate_prob, segs=segs)

    '''
    ###############################################################################################

    # check rotation and deformation are the same with and without rotation.

    win = [2] * 3
    NCC_metric = NCC(win).loss # radius = 2 or window size = 5

    vol_bl = vol_bl.to('cuda')
    vol_fu = vol_fu.to('cuda')
    vol_def = vol_def.to('cuda')

    grid = my_create_pytorch_grid(vol_bl.shape).to('cuda')
    # grid.shape: [2, 84, 121, 114, 3]

    I_res = F.grid_sample(vol_fu, grid + vol_def, align_corners=False)

    # I_res = my_apply_warp(vol_bl, vol_fu, vol_def)

    NCC_img1_img2 = NCC_metric(vol_bl, vol_fu)
    NCC_img1_warp = NCC_metric(vol_bl, I_res)
    NCC_warp_img2 = NCC_metric(I_res, vol_fu)
    print("before rotation NCC_before_reg, NCC_after_reg, NCC_for_reg = ", NCC_img1_img2, NCC_img1_warp, NCC_warp_img2)

    # compare the intensity difference before rotating, of the baseline image and the warped image
    # and the intensity difference after rotating, of the baseline image and warped image.

    if segs:
        vol_bl2, vol_fu2, vol_seg2, vol_def2 = image_list2
    else:
        vol_bl2, vol_fu2, vol_def2 = image_list2

    vol_bl2 = vol_bl2.to('cuda')
    vol_fu2 = vol_fu2.to('cuda')
    vol_def2 = vol_def2.to('cuda')

    grid = my_create_pytorch_grid(vol_bl2.shape).to('cuda')
    # grid.shape: [2, 84, 121, 114, 3]

    I_res2 = F.grid_sample(vol_fu2, grid + vol_def2, align_corners=False)

    # I_res = my_apply_warp(vol_bl, vol_fu, vol_def)

    NCC_img1_img2 = NCC_metric(vol_bl2, vol_fu2)
    NCC_img1_warp = NCC_metric(vol_bl2, I_res2)
    NCC_warp_img2 = NCC_metric(I_res2, vol_fu2)
    print("after rotation NCC_before_reg, NCC_after_reg, NCC_for_reg = ", NCC_img1_img2, NCC_img1_warp, NCC_warp_img2)
    print()

    diff = abs(vol_bl - I_res)
    diff_aug2 = abs(vol_bl2 - I_res2)

    diff_aug2_back = data_aug_gpu.my_transform_image_pytorch(vol_bl, diff_aug2, np.linalg.inv(T_A), T_b,
                                            mode='bilinear', padding_mode='zeros', T_grid=None)

    diff_crop = diff[:, :, 20:-20, 20:-20, 20:-20]
    diff_aug2_back_crop = diff_aug2_back[:, :, 20:-20, 20:-20, 20:-20]

    ddiff = abs(diff_crop - diff_aug2_back_crop)
    print("mean difference after the diff_rot is rotated back", abs(diff_crop - diff_aug2_back_crop).mean() )
    print("max difference after the diff_rot is rotated back", abs(diff_crop - diff_aug2_back_crop).max() )

   #### plot jacobian determinant and attention map

    # setup the figure
    fig = plt.figure()

    # show first image
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("diff_crop")
    # plt.imshow(pred[0, 24, :, :].detach().cpu().numpy(), cmap=plt.cm.Reds, alpha=.5)
    plt.imshow(diff_crop[0, 0, 24, :, :].detach().cpu().numpy(), cmap=plt.cm.gray)

    # show first image
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("diff_aug2_back_crop")
    # plt.imshow(pred[0, 24, :, :].detach().cpu().numpy(), cmap=plt.cm.Reds, alpha=.2)
    plt.imshow(diff_aug2_back_crop[0, 0, 24, :, :].detach().cpu().numpy(), cmap=plt.cm.gray)
    # plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("diff between first two")
    plt.imshow(ddiff[0, 0, 24, :, :].detach().cpu().numpy(), cmap=plt.cm.gray)
    # plt.axis("off")

    plt.savefig("/data/mengjin/Longi_T1_Aim2/vxm_attention_2_ALOHA_20231030/observe_diff_rotation.png")

    # # show the images
    # plt.show()
    # plt.waitforbuttonpress()

    print("difference after the diff_rot is rotated back", abs(diff - diff_aug2_back).mean() )

    image_list1 = image_list2

    #####################################################################################################3
    '''

    # print("vol_bl.shape", vol_bl.shape)
    # print("vol_def.shape", vol_def.shape)
    if 'flip' in augment:
        image_list1 = data_aug_gpu.randomFlip3d(image_list1)

    # Random 3D rotate image
    # print("len(image_list1) = ", len(image_list1))
    # print("dim of deformation field", vol_def.shape)

    if 'random_crop' in augment:
        if segs:
            image_list1, crop_pos = data_aug_gpu.randomCropBySeg3d(image_list1, output_size)
        else:
            # previously, for cropping we should read segmentation image: randomCropBySeg3d()
            # now, we randomly crop to the desired size
            image_list1, crop_pos = data_aug_gpu.randomCrop3d(image_list1, output_size)
    elif 'fixed_crop' in augment:
        # assuming the last dimension is the deformation field dimension
        image_list1, crop_pos = data_aug_gpu.fixedCrop3d(image_list1, output_size)

    if segs:
        vol_bl = image_list1[0]
        vol_fu = image_list1[1]
        vol_seg = image_list1[2]
        vol_def = image_list1[3]
    else:
        vol_bl = image_list1[0]
        vol_fu = image_list1[1]
        vol_def = image_list1[2]

    if add_feat_axis:
        vol_bl = vol_bl[..., np.newaxis]
        vol_fu = vol_fu[..., np.newaxis]
        if segs:
            vol_seg = vol_seg[..., np.newaxis]


    # print("bl shape after aug", vol_bl.shape)
    # print("def shape after aug", vol_def.shape)

    if segs:
        return vol_bl, vol_fu, crop_pos, vol_seg, vol_def
    else:
        return vol_bl, vol_fu, crop_pos, vol_def


def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            # affine = np.array([[-1, 0, 0, 0],  # nopep8
            #                    [0, 0, 1, 0],  # nopep8
            #                    [0, -1, 0, 0],  # nopep8
            #                    [0, 0, 0, 1]], dtype=float)  # nopep8
            # pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            # affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
            affine = np.eye(4) # now using the identity matrix for simulated data
            nib.save(nib.Nifti1Image(array, affine), filename)
        elif len(affine) == 3 and array.ndim >= 3:
            aff = affine[0]
            orig_shape = affine[1]
            crop_pos = affine[2]
            arr = restore_original_image(array, orig_shape, crop_pos)
            nib.save(nib.Nifti1Image(arr, aff), filename)
        else:
            nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)
    

def save_warpfile(array, filename, sform_matrix, affine=None):
    """
    Saves a warp to nii, nii.gz, or npz format.

    Parameters:
        array: The warp to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """

    if filename.endswith(('.nii', '.nii.gz')):
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            # affine = np.array([[-1, 0, 0, 0],  # nopep8
            #                    [0, 0, 1, 0],  # nopep8
            #                    [0, -1, 0, 0],  # nopep8
            #                    [0, 0, 0, 1]], dtype=float)  # nopep8
            # pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            # affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
            affine = np.eye(4) # now using the identity matrix for simulated data
            #TODO add here convert deformation field from pytorch coordinate system to sform system
            nib.save(nib.Nifti1Image(array, affine), filename)
        elif len(affine) == 3 and array.ndim >= 3:
            aff = affine[0]
            orig_shape = affine[1]
            crop_pos = affine[2]
            arr = restore_original_image(array, orig_shape, crop_pos)
            nib.save(nib.Nifti1Image(arr, aff), filename)
        else:
            nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def restore_original_image(array, orig_shape, crop_pos):
    image_shape = array.shape[-3:]
    orig_shape = orig_shape[-3:]

    big_shape = np.max([image_shape, orig_shape], axis=0)
    # a big shape to maintain all data, and then crop
    arr1 = np.zeros(big_shape)
    # if max(crop_pos) > 0:
    #     print("here, max(crop_pos) > 0")
    start10 = max(crop_pos[0], 0)
    start11 = max(crop_pos[1], 0)
    start12 = max(crop_pos[2], 0)
    arr1[start10:start10 + image_shape[0],
         start11:start11 + image_shape[1],
         start12:start12 + image_shape[2]] = array

    # This part might be wrong because there's no case crop_pos is positive.
    # So no testing data.
    start20 = -min(crop_pos[0], 0)
    start21 = -min(crop_pos[1], 0)
    start22 = -min(crop_pos[2], 0)
    arr2 = arr1[start20:start20 + orig_shape[0],
                start21:start21 + orig_shape[1],
                start22:start22 + orig_shape[2]]

    if arr2.shape != tuple(orig_shape):
        print("didn't map to original shape")

    return arr2


# def pad(array, shape):
#     """
#     Zero-pads an array to a given shape. Returns the padded array and crop slices.
#     """
#     if array.shape == tuple(shape):
#         return array, ...

#     padded = np.zeros(shape, dtype=array.dtype)
#     offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
#     slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
#     padded[slices] = array

#     return padded, slices

    """
    Filters given volumes to only include given labels, all other voxels are set to 0.
    """
    mask = np.zeros(atlas_vol.shape, 'bool')
    for label in labels:
        mask = np.logical_or(mask, atlas_vol == label)
    return atlas_vol * mask


def get_moving_volume(moving_segmentation):
    # get sum of positive labels multiplied by resolution
    if len(moving_segmentation.shape) == 5: # 3d
        return torch.sum(moving_segmentation, dim=(1, 2, 3, 4))
    else:
        # do not multiply by resolution on 2D images * 0.5 * 0.6
        return torch.sum(moving_segmentation, dim=(1, 2, 3))


def get_moved_volume(moving_segmentation, jdet):
    # get sum of positive labels * Jac * resolution
    if len(moving_segmentation.shape) == 5:
        return torch.sum(torch.multiply(moving_segmentation, jdet), dim=(1, 2, 3, 4)) * 0.5 * 0.6
    else:
        return torch.sum(torch.multiply(moving_segmentation, jdet), dim=(1, 2, 3))


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def jacobian_determinant_batch(disp):
    """
    jacobian determinant of a batch of displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    batchsize = disp.shape[0]
    volshape = disp.shape[2:]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = torch.from_numpy(np.stack(grid_lst, len(volshape))).cuda()

    Jdet = torch.zeros(batchsize, 1, *volshape).cuda()
    if nb_dims == 3:
        disp = disp.permute(0, 2, 3, 4, 1)
    elif nb_dims == 2:
        disp = disp.permute(0, 2, 3, 1)
    # compute gradients
    for i in range(batchsize):
        J = torch.gradient(disp[i] + grid)

        # 3D glow
        if nb_dims == 3:
            dx = J[0]
            dy = J[1]
            dz = J[2]

            # compute jacobian components
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

            Jdet[i, 0, ...] = Jdet0 - Jdet1 + Jdet2

        else:  # must be 2

            dfdx = J[0]
            dfdy = J[1]

            Jdet[i, 0, ...] =  dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

    return Jdet


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch, normalize = True):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
             normalize (bool): whether normalize the image
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch, normalize=normalize)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch, normalize = True):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_batch(self, name, batch, image_names = None, normalize = True):
        if image_names:
            tag_template = '{}/{}'
        else:
            tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):

                    if image_names:
                        image_name = "/".join(image_names[batch_idx].split("/")[-3:])
                        tag = tag_template.format(name, image_name)
                    else:
                        tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)

                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    if normalize:
                        tagged_images.append((tag, self._normalize_img(img)))
                    else:
                        tagged_images.append((tag, img))
        else:
            # batch hafrom sklearn.decomposition import PCAs no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):

                if image_names:
                    image_name = "/".join(image_names[batch_idx].split("/")[-3:])
                    tag = tag_template.format(name, image_name)
                else:
                    tag = tag_template.format(name, batch_idx, 0, slice_idx)

                img = batch[batch_idx, slice_idx, ...]
                if normalize:
                    tagged_images.append((tag, self._normalize_img(img)))
                else:
                    tagged_images.append((tag, img))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def get_tensorboard_formatter():
    return DefaultTensorboardFormatter()


def get_logger(name, log_file='log.txt', level=logging.INFO):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')

    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Logging to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_epoch_lr(writer, epoch, optimizer, num_iterations):
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('z_learning_rate', lr, num_iterations)
    writer.add_scalar('z_epoch', epoch, num_iterations)


def log_stats(writer, phase, total_loss, loss_list, acc, num_iterations):
    tag_value = {
        f'{phase}_loss_avg': total_loss.avg,
        f'{phase}_acc_hippo_sto1': acc[0].avg,
        f'{phase}_acc_sulcus_sto1': acc[1].avg,
        f'{phase}_acc_hippo_sto2': acc[2].avg,
        f'{phase}_acc_sulcus_sto2': acc[3].avg,
        f'{phase}_acc_hippo_risi': acc[4].avg,
        f'{phase}_acc_sulcus_risi': acc[5].avg,
    }
    for n, subloss in enumerate(loss_list):
        tag_value[f'{phase}_subloss_{n}'] = subloss.avg

    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, num_iterations)


def log_images_subtle(writer, sample, out_seg1, num_iterations, tensorboard_formatter, out_seg2=None):
    # for images, log images to tensorboardX
    nums_display = 4 # was 10
    inputs_map = {
        'moving1':          sample["imgs_bl1"][0:nums_display, 0, :], 
        # input bl and fu at the same time
        'fixed1':           sample["imgs_fu1"][0:nums_display, 0, :],
        'moving2':          sample["imgs_bl2"][0:nums_display, 0, :], 
        'fixed2':           sample["imgs_fu2"][0:nums_display, 0, :],
        'seg_bg1':          out_seg1[0:nums_display, 0, :],
        'seg_hippo1':       out_seg1[0:nums_display, 1, :],
        'seg_sulcus1':      out_seg1[0:nums_display, 2, :],
    }
    if out_seg2 != None:
        inputs_map['seg_bg2'] = out_seg2[0:nums_display, 0, :],
        inputs_map['seg_hippo2'] = out_seg2[0:nums_display, 1, :],
        inputs_map['seg_sulcus2'] = out_seg2[0:nums_display, 2, :],
    if len(sample["imgs_seg1"].shape) > 0:
        inputs_map['segs_gt1'] = sample["imgs_seg1"][0:nums_display, 0, :]
        inputs_map['segs_gt2'] = sample["imgs_seg2"][0:nums_display, 0, :]
    img_sources = {}
    for name, batch in inputs_map.items():
        if isinstance(batch, list) or isinstance(batch, tuple):
            for i, b in enumerate(batch):
                img_sources[f'{name}{i}'] = b.data.cpu().numpy()
        else:
            img_sources[name] = batch.data.cpu().numpy()
    for name, batch in img_sources.items():
        for tag, image in tensorboard_formatter(name, batch, normalize=True): # image_names
            writer.add_image(tag, image, num_iterations, dataformats='CHW')


def train(train_loader, model, model_dir, 
          losses, 
          optimizer, 
          epoch, 
          writer, 
          tensorboard_formatter, 
          log_after_iters, 
          logger, 
          device, 
          risi_loss_weight = 1, 
          save_image = True, 
          save_image_freq = 100, 
          epochs = 100,):
    
    train_epoch_loss = []
    train_epoch_total_acc = []
    train_epoch_total_loss = RunningAverage()

    for n in range(len(losses)):
        train_epoch_loss.append(RunningAverage())
        train_epoch_total_acc.append(RunningAverage())

    epoch_step_time = []

    for i, sample in enumerate(train_loader):
        print(i, "new batch!",  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        step_start_time = time.time()

        if "imgs_seg1" in sample and "imgs_seg2" in sample:
            sample["imgs_seg1"] = sample["imgs_seg1"].squeeze(1).to(device).float()
            sample["imgs_seg2"] = sample["imgs_seg2"].squeeze(1).to(device).float()

        sample["imgs_bl1"] = sample["imgs_bl1"].squeeze(1).to(device).float()
        sample["imgs_bl2"] = sample["imgs_bl2"].squeeze(1).to(device).float()
        sample["imgs_fu1"] = sample["imgs_fu1"].squeeze(1).to(device).float()
        sample["imgs_fu2"] = sample["imgs_fu2"].squeeze(1).to(device).float()

        sample["imgs_warp1"] = sample["imgs_warp1"].squeeze(1).to(device).float()
        sample["imgs_warp2"] = sample["imgs_warp2"].squeeze(1).to(device).float()

        sample["label_date_diff1"] = sample["label_date_diff1"].clone().detach().to(device).float()
        sample["label_date_diff2"] = sample["label_date_diff2"].clone().detach().to(device).float() # date_diff_label
        sample["label_time_interval"] = sample["label_time_interval"].clone().detach().to(device).float()

        batch_size = sample["imgs_bl1"].shape[0]

        # run inputs through the model to produce a warped image and flow field
        # prediction: volume difference between image 1 and 2

        out_seg1, out_seg2, moved_img1, moved_img2, vol_diff_label = model(sample["imgs_bl1"],
                                                    sample["imgs_fu1"],
                                                    sample["imgs_bl2"],
                                                    sample["imgs_fu2"],
                                                    sample["imgs_warp1"],
                                                    sample["imgs_warp2"],
                                                    registration=True)
        attention_pred1 = torch.argmax(out_seg1, dim=1)
        attention_pred2 = torch.argmax(out_seg2, dim=1)

        loss = 0
        for n, loss_function in enumerate(losses):

            if n < 2:
                # loss 0: ce for scan temporal order for the first pair
                curr_loss = loss_function(vol_diff_label[n], sample["label_date_diff1"].long()) 
                date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                train_acc = torch.mean((date_diff_pred == sample["label_date_diff1"]).double())

                train_epoch_loss[n].update(curr_loss.item(), batch_size)
                train_epoch_total_acc[n].update(train_acc.item(), batch_size)
                loss += curr_loss

            elif n < 4:
                # loss 1: ce for scan temporal order for the second pair
                curr_loss = loss_function(vol_diff_label[n], sample["label_date_diff2"].long())
                date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                train_acc = torch.mean((date_diff_pred == sample["label_date_diff2"]).double())

                train_epoch_loss[n].update(curr_loss.item(), batch_size)
                train_epoch_total_acc[n].update(train_acc.item(), batch_size)
                loss += curr_loss

            else:
                # loss 2: mse for RISI
                curr_loss = loss_function(vol_diff_label[n], sample["label_time_interval"].long())
                train_epoch_loss[n].update(curr_loss.item(), batch_size)

                label_time_interval_pred = torch.argmax(vol_diff_label[n], dim=1)
                train_acc = torch.mean((label_time_interval_pred == sample["label_time_interval"]).double())
                train_epoch_total_acc[n].update(train_acc.item(), batch_size)

                loss += risi_loss_weight * curr_loss


        train_epoch_total_loss.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        # print(torch.autograd.grad(loss, vol_diff_label[5]))
        optimizer.step()
        # model.UNet3D_Seg.decoder1.conv_block.conv1.conv.weight

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

        # log training information
        # iterations =  epoch + i / len(train_loader)
        # When logging training, epoch in training can be seen when reducing the last 3 digits.
        iterations =  int((epoch + i / len(train_loader)) * 1000)
        if iterations % log_after_iters == 0:
            loss_text = ""
            for n in range(len(losses)):
                loss_text = loss_text + f'{train_epoch_loss[n].avg},'
            logger.info(
                f'Training stats. Epoch: {epoch + 1}/{epochs}. ' +
                f'Step: {i}/{len(train_loader)} ' +
                f'Loss: {train_epoch_total_loss.avg} (' + loss_text[:-1] + ') ' +
                f'Acc hippo1: {train_epoch_total_acc[0].avg} ' +
                f'Acc sulcus1: {train_epoch_total_acc[1].avg} ' +
                f'Acc hippo2: {train_epoch_total_acc[2].avg} ' +
                f'Acc sulcus2: {train_epoch_total_acc[3].avg} '
                f'Acc hippo RISI: {train_epoch_total_acc[4].avg} ' +
                f'Acc sulcus RISI: {train_epoch_total_acc[5].avg} ' +
                f'Loss ratio_hippo: {train_epoch_loss[4].avg} ' +
                f'Loss ratio_sulcus: {train_epoch_loss[5].avg} '
                # f'Global Reg Bias: {model.beta} '
                )
            log_stats(writer, 'train', train_epoch_total_loss, train_epoch_loss, train_epoch_total_acc, iterations)
            # log: moving, fixed, attention (seg), Jacobian, deformation (flow),

            # out_seg, moved_img, warp, vol_diff_label,
            log_images_subtle(writer, sample, out_seg1, iterations, \
                                tensorboard_formatter, out_seg2)
            # parameters: writer, inputs, prediction, diff, num_iterations

        if save_image and iterations % save_image_freq == 0: # save_image_freq = 100
            
            # save a random selection of (3) images at a time
            num_samples = len(sample["label_date_diff1"])
            for i in random.sample(range(num_samples), min(num_samples, 3)):

                moving_name1 = sample["bl_fname1"][i].split("/")[-1]
                moving_name2 = sample["bl_fname2"][i].split("/")[-1]

                subjectID = moving_name1[:10]
                stage = sample["bl_fname1"][i].split("/")[-2]
                side = moving_name1.split("_")[-3]
                bl_time = moving_name1[11:21]
                fu_time = moving_name1[22:32]

                moving_name1 = model_dir + "/" + str(iterations) + "__" + moving_name1
                fixed_name1 = moving_name1.replace('bl_to_hw', "fu_to_hw", 1)
                moved_name1 = moving_name1.replace('.nii.gz', "_moved.nii.gz", 1)
                warp_name1 = moving_name1.replace('.nii.gz', "_warp.nii.gz", 1)
                moving_seg_name1 = moving_name1.replace('bl_to_hw', "seg_to_hw", 1)

                moving_name2 = model_dir + "/" + str(iterations) + "__" + moving_name2
                fixed_name2 = moving_name2.replace('bl_to_hw', "fu_to_hw", 1)
                moved_name2 = moving_name2.replace('.nii.gz', "_moved.nii.gz", 1)
                warp_name2 = moving_name2.replace('.nii.gz', "_warp.nii.gz", 1)
                moving_seg_name2 = moving_name2.replace('bl_to_hw', "seg_to_hw", 1)
                # gather information of subjects

                # save moving image
                save_volfile(sample["imgs_bl1"][i, 0, :].detach().cpu().numpy().squeeze(), moving_name1)
                save_volfile(sample["imgs_bl2"][i, 0, :].detach().cpu().numpy().squeeze(), moving_name2)

                # save fixed image
                save_volfile(sample["imgs_fu1"][i, 0, :].detach().cpu().numpy().squeeze(), fixed_name1)
                save_volfile(sample["imgs_fu2"][i, 0, :].detach().cpu().numpy().squeeze(), fixed_name2)

                # save attention image
                moving_attention_name1 = moving_name1.replace('.nii.gz', "_attention.nii.gz", 1)
                save_volfile(attention_pred1[i, :].detach().cpu().numpy().squeeze().astype(float), moving_attention_name1)

                moving_attention_name2 = moving_name2.replace('.nii.gz', "_attention.nii.gz", 1)
                save_volfile(attention_pred2[i, :].detach().cpu().numpy().squeeze().astype(float), moving_attention_name2)

                # # save jacobian determinant
                # moving_jdet_name1 = moving_name1.replace('.nii.gz', "_jdet.nii.gz", 1)
                # save_volfile(jdet1[i, 0, :].detach().cpu().numpy().squeeze(), moving_jdet_name1)
                
                # moving_jdet_name2 = moving_name2.replace('.nii.gz', "_jdet.nii.gz", 1)
                # save_volfile(jdet1[i, 0, :].detach().cpu().numpy().squeeze(), moving_jdet_name2)

                # save seg image before registration
                save_volfile(sample["imgs_seg1"][i, 0, :].detach().cpu().numpy().squeeze(), moving_seg_name1)
                save_volfile(sample["imgs_seg2"][i, 0, :].detach().cpu().numpy().squeeze(), moving_seg_name2)

                # save moved image
                save_volfile(moved_img1[i, 0, :].detach().cpu().numpy().squeeze(), moved_name1)
                save_volfile(moved_img2[i, 0, :].detach().cpu().numpy().squeeze(), moved_name2)

                # save warp image
                save_volfile(sample["imgs_warp1"][i, :].detach().cpu().permute(1, 2, 3, 0).numpy().squeeze(), warp_name1)
                save_volfile(sample["imgs_warp2"][i, :].detach().cpu().permute(1, 2, 3, 0).numpy().squeeze(), warp_name2)


def validate(eval_loader, model, 
                losses, 
                optimizer, 
                epoch, 
                writer, 
                logger, 
                device, 
                len_train_loader, 
                risi_loss_weight = 1, 
                epochs = 100):

    val_epoch_loss = []
    val_epoch_total_acc = []
    val_epoch_total_loss = RunningAverage()

    for n in range(len(losses)):
        val_epoch_loss.append(RunningAverage())
        val_epoch_total_acc.append(RunningAverage())

    with torch.no_grad():

        for i, sample in enumerate(eval_loader):
            print(i, "new batch!",  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            step_start_time = time.time()

            if "imgs_seg1" in sample and "imgs_seg2" in sample:
                sample["imgs_seg1"] = sample["imgs_seg1"].squeeze(1).to(device).float()
                sample["imgs_seg2"] = sample["imgs_seg2"].squeeze(1).to(device).float()

            sample["imgs_bl1"] = sample["imgs_bl1"].squeeze(1).to(device).float()
            sample["imgs_bl2"] = sample["imgs_bl2"].squeeze(1).to(device).float()
            sample["imgs_fu1"] = sample["imgs_fu1"].squeeze(1).to(device).float()
            sample["imgs_fu2"] = sample["imgs_fu2"].squeeze(1).to(device).float()
    
            sample["imgs_warp1"] = sample["imgs_warp1"].squeeze(1).to(device).float()
            sample["imgs_warp2"] = sample["imgs_warp2"].squeeze(1).to(device).float()

            sample["label_date_diff1"] = sample["label_date_diff1"].clone().detach().to(device).float()
            sample["label_date_diff2"] = sample["label_date_diff2"].clone().detach().to(
                device).float()  # date_diff_label
            sample["label_time_interval"] = sample["label_time_interval"].clone().detach().to(
                device).float()  # label_time_interval
            
            batch_size = sample["imgs_bl1"].shape[0]

            # run inputs through the model to produce a warped image and flow field
            # prediction: volume difference between image 1 and 2
            out_seg1, out_seg2, moved_img1, moved_img2, vol_diff_label = model(sample["imgs_bl1"],
                                                        sample["imgs_fu1"],
                                                        sample["imgs_bl2"],
                                                        sample["imgs_fu2"],
                                                        sample["imgs_warp1"],
                                                        sample["imgs_warp2"],
                                                        registration=True)

            attention_pred1 = torch.argmax(out_seg1, dim=1)
            attention_pred2 = torch.argmax(out_seg2, dim=1)

            # calculate total loss
            loss_val = 0
            for n, loss_function in enumerate(losses):

                if n < 2:
                    # loss 0: ce for scan temporal order for the first pair
                    curr_loss_val = loss_function(vol_diff_label[n], sample[
                        "label_date_diff1"].long())
                    date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                    val_acc = torch.mean((date_diff_pred == sample["label_date_diff1"]).double())

                    val_epoch_loss[n].update(curr_loss_val.item(), batch_size)
                    val_epoch_total_acc[n].update(val_acc.item(), batch_size)

                    loss_val += curr_loss_val

                elif n < 4:
                    # loss 1: ce for scan temporal order for the second pair
                    curr_loss_val = loss_function(vol_diff_label[n], sample[
                        "label_date_diff2"].long())  # * train_batch_size / math.prod(inshape)
                    date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                    val_acc = torch.mean((date_diff_pred == sample["label_date_diff2"]).double())

                    val_epoch_loss[n].update(curr_loss_val.item(), batch_size)
                    val_epoch_total_acc[n].update(val_acc.item(), batch_size)

                    loss_val += curr_loss_val

                else:
                    # loss 2: mse for RISI
                    curr_loss_val = loss_function(vol_diff_label[n], sample[
                        "label_time_interval"].long())  # * train_batch_size / math.prod(inshape)
                    val_epoch_loss[n].update(curr_loss_val.item(), batch_size)

                    label_time_interval_pred = torch.argmax(vol_diff_label[n], dim=1)
                    val_acc = torch.mean((label_time_interval_pred == sample["label_time_interval"]).double())
                    val_epoch_total_acc[n].update(val_acc.item(), batch_size)

                    loss_val += risi_loss_weight * curr_loss_val

    val_epoch_total_loss.update(loss_val.item(), batch_size)

    # log stats
    loss_text = ""
    for n in range(len(losses)):
        loss_text = loss_text + f'{val_epoch_loss[n].avg},'
    logger.info(
        f'Validation stats. Epoch: {epoch + 1}/{epochs}. ' +
        f'Loss: {val_epoch_total_loss.avg} (' + loss_text[:-1] + ') ' +
        f'Acc hippo1: {val_epoch_total_acc[0].avg} ' +
        f'Acc sulcus1: {val_epoch_total_acc[1].avg} ' +
        f'Acc hippo2: {val_epoch_total_acc[2].avg} ' +
        f'Acc sulcus2: {val_epoch_total_acc[3].avg} '
        f'Acc hippo RISI: {val_epoch_total_acc[4].avg} ' +
        f'Acc sulcus RISI: {val_epoch_total_acc[5].avg} ' +
        f'Loss ratio_hippo: {val_epoch_loss[4].avg} ' +
        f'Loss ratio_sulcus: {val_epoch_loss[5].avg} '
        )
    log_stats(writer, 'val', val_epoch_total_loss, val_epoch_loss, val_epoch_total_acc,
            len_train_loader * (epoch + 1))
    log_epoch_lr(writer, epoch, optimizer, len_train_loader * (epoch + 1))

    return val_epoch_total_loss


def validate_pair(csv_name, test_loader, model, 
                  model_dir, 
                  device,
                  inshape,
                  logger):
    
    epoch_step_time = []
    val_epoch_total_acc1 = RunningAverage()
    val_epoch_total_acc2 = RunningAverage()

    with torch.no_grad():
        with open(csv_name, 'a', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(["moving_fname", "fixed_fname",
                            "subjectID", "side", "stage", "bl_time", "fu_time",
                            "date_diff", "label_date_diff",
                            "date_diff_pred1", "date_diff_pred2",
                            "date_diff_acc1", "date_diff_acc2",
                            # "vol_diff_ratio_hippo", "vol_diff_ratio_sulcus", 
                            "bl_hippo_volume", "fu_hippo_volume",
                            "bl_sulcus_volume", "fu_sulcus_volume",
                            "hippo_attn_voxel", "sulcus_attn_voxel",])
    

            for i, sample in enumerate(test_loader):
                print(i, "new batch!",  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                step_start_time = time.time()

                # generate inputs (and true outputs) and convert them to tensors
                sample["imgs_seg"] = sample["imgs_seg"].squeeze(1).to(device).float() # .permute(0, 4, 1, 2, 3)
                sample["imgs_bl"] = sample["imgs_bl"].squeeze(1).to(device).float() 
                sample["imgs_fu"] = sample["imgs_fu"].squeeze(1).to(device).float() 
                sample["imgs_warp"] = sample["imgs_warp"].squeeze(1).to(device).float() 

                sample["label_date_diff"] = sample["label_date_diff"].clone().detach().to(device).float()

                batch_size = sample["imgs_bl"].shape[0]

                x = torch.cat([sample["imgs_bl"], sample["imgs_fu"] ], dim=1)
                out_seg = model.UNet3D_Seg(x)
                out_seg, moved_img, volume_diff, volume, warped_volume, jdet = model.attention(sample["imgs_bl"],
                                sample["imgs_fu"],
                                sample["imgs_warp"],
                                out_seg,
                                registration=True)

                volume_diff_divide_hippo, volume_diff_divide_sulcus = volume_diff
                hippo_volume, sulcus_volume = volume
                warped_hippo_volume, warped_sulcus_volume = warped_volume

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


                val_acc1 = torch.mean((date_diff_pred1 == sample["label_date_diff"]).double())
                val_epoch_total_acc1.update(val_acc1.item(), batch_size)

                val_acc2 = torch.mean((date_diff_pred2 == sample["label_date_diff"]).double())
                val_epoch_total_acc2.update(val_acc2.item(), batch_size)

                logger.info(
                    f'Validation stats: '
                    f'Acc hippo: {val_epoch_total_acc1.avg} '
                    f'Acc sulcus: {val_epoch_total_acc2.avg} '
                    )
                
                
                for i in range(len(sample["label_date_diff"])):

                    moving_name = sample["bl_fname"][i].split("/")[-1]
                    # print("moving_name", moving_name)
                    moving_seg_name = model_dir + "/" + moving_name.replace('bl_to_hw', "seg_to_hw", 1)

                    subjectID = moving_name[:10]
                    #TODO
                    stage = sample["stage"][i]
                    side = sample["side"][i]
                    bl_time = moving_name[11:21]
                    fu_time = moving_name[22:32]

                    moving_name = model_dir + "/" + moving_name
                    fixed_name = moving_name.replace('bl_to_hw', "fu_to_hw", 1)
                    moved_name = moving_name.replace('.nii.gz', "_moved.nii.gz", 1)
                    warp_name = moving_name.replace('.nii.gz', "_warp.nii.gz", 1)

                    # gather information of subjects
                    row = [moving_name, fixed_name,
                            subjectID, side, stage, bl_time, fu_time,
                            sample["date_diff"][i].item(), sample["label_date_diff"][i].item(),
                            date_diff_pred1[i].item(), date_diff_pred2[i].item(),
                            int(sample["label_date_diff"][i].item() == date_diff_pred1[i].item()), int(sample["label_date_diff"][i].item() == date_diff_pred2[i].item()),
                            #    vol_diff_label[0][i][0].item(), vol_diff_label[1][i][0].item(),
                            hippo_volume[i].item(), warped_hippo_volume[i].item(),
                            sulcus_volume[i].item(), warped_sulcus_volume[i].item(),
                            hippo_attn_vox[i].item(), sulcus_attn_vox[i].item(),
                            ]
                    
                    # print(i)
                    # print("vol_diff_label                                                 ", 
                    #       vol_diff_label[0][i][0].item(), vol_diff_label[1][i][0].item())
                    # print("hippo_volume.item(), warped_hippo_volume.item()                ", 
                    #       hippo_volume[i].item(), warped_hippo_volume[i].item())
                    # print("sulcus_volume.item(), warped_sulcus_volume.item()              ", 
                    #       sulcus_volume[i].item(), warped_sulcus_volume[i].item())
                    # print("hippo_attn_vox.item(), sulcus_attn_vox.item()                  ",
                    #       hippo_attn_vox[i].item(), sulcus_attn_vox[i].item())

                    wr.writerow(row)
                    # print(row)

                
                    # save example images, sample the last images of each batch
                    affine_trans = [sample["affines"][i], 
                                    [dim[i].item() for dim in sample["orig_shapes"]],
                                    [dim[i].item() for dim in sample["crop_posns"]]]

                    # save bl image
                    shutil.copyfile(sample["bl_fname"][i], moving_name) 

                    # save attention image
                    moving_attention_name = moving_name.replace('.nii.gz', "_attention.nii.gz", 1)
                    save_volfile(attention_pred[i, :].detach().cpu().numpy().squeeze().astype(float),
                                       moving_attention_name, affine_trans)

                    # save position mask
                    moving_pos_mask = moving_name.replace('.nii.gz', "_position_mask.nii.gz", 1)
                    save_volfile(np.ones(inshape), moving_pos_mask, affine_trans)

                    # save jacobian determinant
                    moving_jdet_name = moving_name.replace('.nii.gz', "_jdet.nii.gz", 1)
                    save_volfile(jdet[i, 0, :].detach().cpu().numpy().squeeze(),
                                        moving_jdet_name, affine_trans)

                    # # save seg image before registration
                    save_volfile(sample["imgs_seg"][i].detach().cpu().numpy().squeeze(), moving_seg_name, affine_trans)

                    # save moved image
                    save_volfile(moved_img[i, 0, :].detach().cpu().numpy().squeeze(),
                                                moved_name, affine_trans)

                    # get compute time
                    epoch_step_time.append(time.time() - step_start_time)
