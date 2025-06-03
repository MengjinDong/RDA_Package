# 3D random thin-plate-spline from Jiancong Wang.
# For an intro to thin-plate-spline, see 30:00 to 38:00
# https://www.youtube.com/watch?v=8zSPmkPqwWs
import scipy.spatial.distance as ssd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from scipy import ndimage
from scipy.stats import ortho_group, special_ortho_group


def Normalize(image_list):

    for i in range(len(image_list)):
        image_list[i] = (image_list[i] - image_list[i].mean()) / max(image_list[i].std(), 1e-5)

    return image_list


def randomErase3d(image_list, EPSILON = 0.5, sl = 0.002, sh = 0.2, r1 = 0.3):

    if random.uniform(0, 1) > EPSILON:
        return image_list

    if len(image_list) <= 2:
        size0 = image_list[0].shape[-3]
        size1 = image_list[0].shape[-2]
        size2 = image_list[0].shape[-1]
    else:

        size0 = min(image_list[0].shape[-3], image_list[-1].shape[-3])
        size1 = min(image_list[0].shape[-2], image_list[-1].shape[-2])
        size2 = min(image_list[0].shape[-1], image_list[-1].shape[-1])

    volume = size0 * size1 * size2

    for attempt in range(100):

        target_volume = random.uniform(sl, sh) * volume
        aspect_ratio1 = random.uniform(r1, 1 / r1)
        aspect_ratio2 = random.uniform(r1, 1 / r1)

        h = int((target_volume * aspect_ratio1 * aspect_ratio2) ** (1./3) )
        w = int((target_volume * aspect_ratio1 / (aspect_ratio2 * aspect_ratio2) ) ** (1./3) )
        l = int((target_volume * aspect_ratio2 / (aspect_ratio1 * aspect_ratio1) ) ** (1./3) )

        if h < size0 and w < size1 and l < size2:
            x1 = random.randint(0, size0 - h)
            y1 = random.randint(0, size1 - w)
            z1 = random.randint(0, size2 - l)

            for i in range(len(image_list)):
                image_list[i][x1:x1 + h, y1:y1 + w, z1:z1 + l] = torch.randn(h, w, l) # Gaussian Distribution
                # image_list[0, x1:x1 + h, y1:y1 + w] = self.mean[1]
                # image_list[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))

            return image_list

    return image_list


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


def my_apply_affine_to_pytorch_grid(A, b, grid):
    px = grid[0,:,:,:,0]
    py = grid[0,:,:,:,1]
    pz = grid[0,:,:,:,2]    
    qx = A[0,0]*px + A[0,1]*py + A[0,2]*pz + b[0]
    qy = A[1,0]*px + A[1,1]*py + A[1,2]*pz + b[1]
    qz = A[2,0]*px + A[2,1]*py + A[2,2]*pz + b[2]
    return torch.stack((qx,qy,qz),3).unsqueeze(0).type(grid.dtype)


def my_transform_image_pytorch(T_ref, T_mov, T_A, T_b, 
                               mode='bilinear', padding_mode='zeros', T_grid=None):
    """
    Apply an affine transform to 3D images represented as PyTorch tensors
    
    :param T_ref: Fixed (reference) image, represented as a 5D tensor
    :param T_mov: Moving image, represented as a 5D tensor
    :param T_A: affine matrix in PyTorch coordinate space, represented as a shape (3,3) tensor
    :param T_b: translation vector in PyTorch coordinate space, represented as a shape (3) tensor
    :param mode: Interpolation mode, see grid_sample
    :param padding_mode: Padding mode, see grid_sample
    :param grid: Optional sampling grid (otherwise will call my_create_pytorch_grid) 
    :returns: Transformed moving image, represented as a 5D tensor
    """
    if T_grid is None:
        T_grid = my_create_pytorch_grid(T_ref.shape, dtype=T_ref.dtype, device=T_ref.device)
        
    T_grid_xform = my_apply_affine_to_pytorch_grid(T_A, T_b, T_grid)

    T_fu_reslice = F.grid_sample(T_mov, T_grid_xform,
                                    mode=mode, padding_mode=padding_mode, 
                                    align_corners=False)
    return T_fu_reslice


def my_transform_warp_pytorch(T_ref, T_warp, T_A, T_b,
                               mode='bilinear', padding_mode='zeros', T_grid=None):
    """
    Apply an affine transform to 3D images represented as PyTorch tensors

    :param T_ref: Fixed (reference) image, represented as a 5D tensor
    # :param T_mov: Moving image, represented as a 5D tensor (after rotation)
    :param T_warp: Warp image, represented as a 5D tensor
    :param T_A: affine matrix in PyTorch coordinate space, represented as a shape (3,3) tensor
    :param T_b: translation vector in PyTorch coordinate space, represented as a shape (3) tensor
    :param mode: Interpolation mode, see grid_sample
    :param padding_mode: Padding mode, see grid_sample
    :param grid: Optional sampling grid (otherwise will call my_create_pytorch_grid)
    :returns: Transformed moving image, represented as a 5D tensor
    """
    if T_grid is None:
        T_grid = my_create_pytorch_grid(T_ref.shape, dtype=T_ref.dtype, device=T_ref.device)

    T_grid_xform = my_apply_affine_to_pytorch_grid(T_A, T_b, T_grid)

    T_warped_def = F.grid_sample(T_warp, T_grid_xform,
                                    mode=mode, padding_mode=padding_mode,
                                    align_corners=False)

    T_warped_def2 = T_warped_def.permute(0, 2, 3, 4, 1) + T_grid_xform
    T_grid_xform2 = my_apply_affine_to_pytorch_grid(np.linalg.inv(T_A), -T_b, T_warped_def2)
    return T_grid_xform2 - T_grid


def randomRotation3d(image_list, max_angle, prob, segs=True):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    image_list: list of images, [bl, fu, seg, def]
    max_angle: `float`. The maximum rotation angle.

    Returns:
    A randomly rotated 3D image and the mask
    """
    # Consider this function being used in multithreading in pytorch's dataloader,
    # if one don't reseed each time this thing is run, the couple worker in pytorch's
    # data worker will produce exactly the same random number and that's no good.

    # bl is the fixed image, image_list[0]
    # fu is the moving image, image_list[1]
    # warp is the deformation field from fu to bl, image_list[-1]

    rotate_prob = np.random.random() < prob
    if rotate_prob > 0:
        # T_A = special_ortho_group.rvs(3)
        T_A = ortho_group.rvs(3)
        T_b = np.array([0, 0, 0])

        I_fix = image_list[0]
        I_mov = image_list[1]
        I_seg = image_list[2]
        I_warp = image_list[-1]
        
        # I_fix.shape [1, 1, 84, 121, 114]
        I_mov_aug = my_transform_image_pytorch(I_fix, I_mov, T_A, T_b,
                                            mode='bilinear', padding_mode='zeros', T_grid=None)

        I_fix_aug = my_transform_image_pytorch(I_fix, I_fix, T_A, T_b,
                                            mode='bilinear', padding_mode='zeros', T_grid=None)

        I_warp_aug = my_transform_warp_pytorch(I_fix, I_warp.permute(0, 4, 1, 2, 3), T_A, T_b,
                                                mode='bilinear', padding_mode='zeros', T_grid=None)

        if segs:
            I_seg_aug = my_transform_image_pytorch(I_fix, I_seg, T_A, T_b,
                                                   mode='bilinear', padding_mode='zeros', T_grid=None)
            I_seg_aug = I_seg_aug > 0.5

            return [I_fix_aug, I_mov_aug, I_seg_aug, I_warp_aug]
        else:
            return [I_fix_aug, I_mov_aug, I_warp_aug]

    return image_list


def randomFlip3d(image_list):
    flip = random.randint(0, 5)  # generate a <= N <= b

    for i in range(len(image_list)):
        if flip == 0:
            image_list[i] = image_list[i][:, :, :, :, ::-1]
        elif flip == 1:
            image_list[i] = image_list[i][:, :, :, ::-1, :]
        elif flip == 2:
            image_list[i] = image_list[i][:, :, ::-1, :, :]

    if flip == 0:
        image_list[-1][:, :, :, :, 0] = - image_list[-1][:, :, :, :, 0]
    elif flip == 1:
        image_list[-1][:, :, :, :, 1] = - image_list[-1][:, :, :, :, 1]
    elif flip == 2:
        image_list[-1][:, :, :, :, 2] = - image_list[-1][:, :, :, :, 2]

    return image_list


def randomCrop3d(image_list, output_size):
    # 3D crop ensuring most area of the mask is inside the cropped image.
    # if dimension of mask is less than required dimension, then random crop outside the mask;
    # if greater, then random crop inside the mask.

    # the last image is the mask

    # print(image_list[0].shape)
    # image_list[2] = ndimage.grey_dilation(image_list[2], size=(5, 5, 5))  # (dilated mask)
    # image_list[2] = ndimage.gaussian_filter(image_list[2], sigma=2)

    a = b = c = []
    if len(a) == 0:  # all labels are zeros
        mask_f = 0
        mask_b = image_list[0].shape[-3]  # the same size as the input image
        mask_u = 0
        mask_d = image_list[0].shape[-2]
        mask_l = 0
        mask_r = image_list[0].shape[-1]

    else:
        mask_f = min(a)  # front
        mask_b = max(a)  # back
        mask_u = min(b)  # up
        mask_d = max(b)  # down
        mask_l = min(c)  # left
        mask_r = max(c)  # right

    mask_t = mask_b - mask_f
    mask_h = mask_d - mask_u
    mask_w = mask_r - mask_l

    t, h, w = image_list[0].shape[-3:]

    # thickness, height, width: desired output size
    # [48, 80, 64]
    thickness, height, width = output_size
    crop_pos_t, crop_pos_h, crop_pos_w = 0, 0, 0

    if t < thickness:
        thick_diff = int(np.floor((thickness - t) / 2))
        for i in range(len(image_list)):
            temp = torch.zeros([1, image_list[i].shape[1], thickness, h, w], device='cuda')
            temp[:, :, thick_diff: thick_diff + t, :, :] = image_list[i]
            image_list[i] = temp
        t = thickness
        crop_pos_t = -thick_diff  #####
    if h < height:
        height_diff = int(np.floor((height - h) / 2))
        for i in range(len(image_list)):
            temp = torch.zeros([1, image_list[i].shape[1], t, height, w], device='cuda')
            temp[:, :, :, height_diff: height_diff + h, :] = image_list[i]
            image_list[i] = temp
        h = height
        crop_pos_h = -height_diff  #####
    if w < width:
        width_diff = int(np.floor((width - w) / 2))
        for i in range(len(image_list)):
            temp = torch.zeros([1, image_list[i].shape[1], t, h, width], device='cuda')
            temp[:, :, :, :, width_diff: width_diff + w] = image_list[i]
            image_list[i] = temp
        w = width
        crop_pos_w = -width_diff  #####

    if mask_t > thickness:
        front = np.random.randint(mask_f, mask_b - thickness)
        crop_pos_t = front  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, front: front + thickness, :, :]
    elif mask_t <= thickness and t > thickness:
        front = np.random.randint(0, min([mask_f, t - thickness]) + 1)
        crop_pos_t = front  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, front: front + thickness, :, :]

    if mask_h > height:
        top = np.random.randint(mask_u, mask_d - height)
        crop_pos_h = top  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, top: top + height, :]
    elif mask_h <= height and h > height:
        top = np.random.randint(0, min([mask_u, h - height]) + 1)
        crop_pos_h = top  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, top: top + height, :]

    if mask_w > width:
        left = np.random.randint(mask_l, mask_r - width)
        crop_pos_w = left  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, :, left: left + width]
    elif mask_w <= width and w > width:
        left = np.random.randint(0, min([mask_l, w - width]) + 1)
        crop_pos_w = left  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, :, left: left + width]

    for i in range(len(image_list)):
        if tuple(image_list[i].shape[-3:]) != output_size:
            print(i)
            print("image_list[i].shape", image_list[i].shape)
        # assert output_image_list[i].shape == output_size

    return image_list, [crop_pos_t, crop_pos_h, crop_pos_w]


def randomCropBySeg3d(image_list, output_size):
    # 3D crop ensuring most area of the mask is inside the cropped image.
    # if dimension of mask is less than required dimension, then random crop outside the mask;
    # if greater, then random crop inside the mask.

    # the second last image is the mask
    mask = image_list[2]

    # the last image is the deformation field
    image_list[-1] = image_list[-1].permute(0, 4, 1, 2, 3)

    # mask = ndimage.grey_dilation(image_list[2], size=(5, 5, 5))  # (dilated mask)
    # mask = ndimage.gaussian_filter(mask, sigma=2)

    # print("running randomCropBySeg3d")
    # for i in range(len(image_list)):
    #     print("image_list[i].shape", image_list[i].shape)

    var = torch.nonzero(mask).detach().cpu().numpy()
    a = var[:, 2]
    b = var[:, 3]
    c = var[:, 4]
    if len(a) == 0:  # all labels are zeros
        mask_f = 0
        mask_b = mask.shape[-3]  # the same size as the input image
        mask_u = 0
        mask_d = mask.shape[-2]
        mask_l = 0
        mask_r = mask.shape[-1]

    else:
        mask_f = min(a)  # front
        mask_b = max(a)  # back
        mask_u = min(b)  # up
        mask_d = max(b)  # down
        mask_l = min(c)  # left
        mask_r = max(c)  # right

    mask_t = mask_b - mask_f
    mask_h = mask_d - mask_u
    mask_w = mask_r - mask_l

    t, h, w = image_list[0].shape[-3:]

    # thickness, height, width: desired output size
    # [48, 80, 64]
    thickness, height, width = output_size
    crop_pos_t, crop_pos_h, crop_pos_w  = 0, 0, 0

    if t < thickness:
        thick_diff = int(np.floor((thickness - t) / 2))
        for i in range(len(image_list)):
            temp = torch.zeros([1, image_list[i].shape[1], thickness, h, w], device='cuda')
            temp[:, :, thick_diff: thick_diff + t, :, :] = image_list[i]
            image_list[i] = temp
        t = thickness
        crop_pos_t = -thick_diff #####
    if h < height:
        height_diff = int(np.floor((height - h) / 2))
        for i in range(len(image_list)):
            temp = torch.zeros([1, image_list[i].shape[1], t, height, w], device='cuda')
            temp[:, :, :, height_diff: height_diff + h, :] = image_list[i]
            image_list[i] = temp
        h = height
        crop_pos_h = -height_diff #####
    if w < width:
        width_diff = int(np.floor((width - w) / 2))
        for i in range(len(image_list)):
            temp = torch.zeros([1, image_list[i].shape[1], t, h, width], device='cuda')
            temp[:, :, :, :, width_diff: width_diff + w] = image_list[i]
            image_list[i] = temp
        w = width
        crop_pos_w = -width_diff #####

    if mask_t > thickness:
        front = np.random.randint(mask_f, mask_b - thickness)
        crop_pos_t = front #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, front: front + thickness, :, :]
    elif mask_t <= thickness and t > thickness:
        front = np.random.randint(0, min([mask_f, t - thickness]) + 1)
        crop_pos_t = front  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, front: front + thickness, :, :]

    if mask_h > height:
        top = np.random.randint(mask_u, mask_d - height)
        crop_pos_h = top #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, top: top + height, :]
    elif mask_h <= height and h > height:
        top = np.random.randint(0, min([mask_u, h - height]) + 1)
        crop_pos_h = top #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, top: top + height, :]

    if mask_w > width:
        left = np.random.randint(mask_l, mask_r - width)
        crop_pos_w = left #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, :, left: left + width]
    elif mask_w <= width and w > width:
        left = np.random.randint(0, min([mask_l, w - width]) + 1)
        crop_pos_w = left  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, :, left: left + width]

    for i in range(len(image_list)):
        if tuple(image_list[i].shape[-3:]) != output_size:
            print(i)
            print("image_list[i].shape", image_list[i].shape)
        # assert output_image_list[i].shape == output_size

    return image_list, [crop_pos_t, crop_pos_h, crop_pos_w]


def fixedCrop3d(image_list, output_size):
    # 3D crop at the center of the image, for the test set.
    # if dimension of image is less than required dimension, then pad the image;
    # if greater, then random crop inside the image.

    # the last image is the mask

    # print(image_list[0].shape)
    image_list[-1] = image_list[-1].permute(0, 4, 1, 2, 3)

    t, h, w = image_list[0].shape[-3:] # original image size

    # thickness, height, width: desired output size
    # [48, 80, 64]
    thickness, height, width = output_size
    crop_pos_t, crop_pos_h, crop_pos_w  = 0, 0, 0

    if t < thickness:
        thick_diff = int(np.floor((thickness - t) / 2))
        for i in range(len(image_list)):
            temp = torch.zeros([1, image_list[i].shape[1], thickness, h, w], device='cuda')
            temp[:, :, thick_diff: thick_diff + t, :, :] = image_list[i]
            image_list[i] = temp
        t = thickness
        crop_pos_t = -thick_diff #####
    if h < height:
        height_diff = int(np.floor((height - h) / 2))
        for i in range(len(image_list)):
            temp = torch.zeros([1, image_list[i].shape[1], t, height, w], device='cuda')
            temp[:, :, :, height_diff: height_diff + h, :] = image_list[i]
            image_list[i] = temp
        h = height
        crop_pos_h = -height_diff #####
    if w < width:
        width_diff = int(np.floor((width - w) / 2))
        for i in range(len(image_list)):
            temp = torch.zeros([1, image_list[i].shape[1], t, h, width], device='cuda')
            temp[:, :, :, :, width_diff: width_diff + w] = image_list[i]
            image_list[i] = temp
        w = width
        crop_pos_w = -width_diff #####

    if t >= thickness:
        front = int(np.floor((t - thickness) / 2))
        crop_pos_t = front #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, front: front + thickness, :, :]

    if h >= height:
        top = int(np.floor((h - height) / 2))
        crop_pos_h = top #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, top: top + height, :]

    if w > width:
        left = int(np.floor((w - width) / 2))
        crop_pos_w = left #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, :, :, left: left + width]

    for i in range(len(image_list)):
        if tuple(image_list[i].shape[-3:]) != output_size:
            print(i)
            print("image_list[i].shape", image_list[i].shape)
        # assert output_image_list[i].shape == output_size

    return image_list, [crop_pos_t, crop_pos_h, crop_pos_w]


class TPSRandomSampler3D(nn.Module):

    def __init__(self, height, width, depth,
                 vertical_points=10, horizontal_points=10, depth_points=10,
                 rotsd=0.0, scalesd=0.0, transsd=0.1, warpsd=(0.001, 0.005),
                 cache_size=1000, cache_evict_prob=0.01, pad=True, device=None):
        super(TPSRandomSampler3D, self).__init__()

        self.input_height = height
        self.input_width = width
        self.input_depth = depth

        self.h_pad = 0
        self.w_pad = 0
        self.d_pad = 0
        if pad:
            self.h_pad = self.input_height // 2
            self.w_pad = self.input_width // 2
            self.d_pad = self.input_depth // 2

        self.height = self.input_height + self.h_pad
        self.width = self.input_width + self.w_pad
        self.depth = self.input_depth + self.d_pad

        self.vertical_points = vertical_points
        self.horizontal_points = horizontal_points
        self.depth_points = depth_points

        self.rotsd = rotsd
        self.scalesd = scalesd
        self.transsd = transsd
        self.warpsd = warpsd
        self.cache_size = cache_size
        self.cache_evict_prob = cache_evict_prob

        # Grid generator. This thing stored the thin-plate-spline kernel given 
        # the height/width/vertical_points/horizontal_points. 
        self.tps = TPSGridGen3D(
            self.height, self.width, self.depth,
            vertical_points, horizontal_points, depth_points)

        self.cache = [None] * self.cache_size if self.cache_size > 0 else []

        self.pad = pad

        self.device = device


    def _sample_grid(self):
        # Returns randomly sampled TPS-grid params of size (Hc*Wc+3)x2.
        # This contains the affine components and random components
        W = sample_tps_w_3D(
            self.vertical_points, self.horizontal_points, self.depth_points,
            self.warpsd,
            self.rotsd, self.scalesd, self.transsd)
        W = torch.from_numpy(W.astype(np.float32))
        # generate grid
        # The forward of gridgen receive the affine and random parameter generated 
        # by the sample_tps_w and returned the perturbed grid
        grid = self.tps(W[None])
        return grid


    # For each sample, sample its own deformation grid
    # This simply cache couple deformed grid. When the cache is full, no new random grid
    # will be generated by just randomly picked one old one.
    def _get_grids(self, batch_size):
        grids = []
        for i in range(batch_size):
            # cache_evict_prob: Has certain probability that new grid get generated.
            save_result = self.cache_size!= 0
            entry = random.randint(0, self.cache_size - 1) if save_result else 0
            if not save_result or self.cache[entry] is None or random.random() < self.cache_evict_prob:
                grid = self._sample_grid()
                
                if save_result:
                    self.cache[entry] = grid
            else:
                grid = self.cache[entry]
            grids.append(grid)
        grids = torch.cat(grids)
        return grids


    def forward(self, input):
        # If device has been specified, ship data to there
        if self.device is not None:
            input = input.to(self.device)

        # get TPS grids
        batch_size = input.size(0)
        grids = self._get_grids(batch_size)
        
        # Ship the grid to the same device as input
        grids = grids.to(input.device)

        # Pad, sample, unpad
        input = F.pad(input, (self.h_pad, self.h_pad, 
                              self.w_pad, self.w_pad,
                              self.d_pad, self.d_pad), mode='replicate')
        
        # The pytorch official implementation of giving a grid and a data, sample position
        input = F.grid_sample(input, grids) 
        # This will sample the grid in input. The grid in normalized coordinate (-1, 1). 
        # This returns the size same as grid. Also it supports 3D image. Heck this is awesome.
        input = F.pad(input, (-self.h_pad, -self.h_pad, 
                              -self.w_pad, -self.w_pad,
                              -self.d_pad, -self.d_pad))
        # Can pad with minus length?! Damn...

        return input

    # This is wrapper function that fit for tensorflow use.
    def forward_py(self, input):
        with torch.no_grad():
            input = torch.from_numpy(input)
            input = input.permute([0, 4, 1, 2, 3]) # [b, h, w, c] -> [b, c, h, w]
            input = self.forward(input)
            input = input.permute([0, 2, 3, 4, 1]) # [b, c, h, w] -> [b, h, w, c]
            input = input.numpy()
            return input
    
    # This is wrapper function that fit for pytorch preprocessing use.
    def forward_torch(self, input, device = None):
        with torch.no_grad():
            input = torch.from_numpy(input)
            
            if device is not None:
                input = input.cuda(device = device)
            input = self.forward(input)
            input = input.cpu().numpy()
            return input


# Grid generator. This thing given the affine and random parameter 
# generate perturbed grid.
# If use a grid as large as the original image, the matrix will be huge. A more 
# reasonable thing to do is to use a downsized grid, apply the W on the downsampled
# grid and upsample it back.            
class TPSGridGen3D(nn.Module):
    def __init__(self, Ho, Wo, Do, Hc, Wc, Dc, scale_factor = 0.2):
        """
        Ho,Wo,Do: height/width/depth of the output tensor (grid dimensions).
        Hc,Wc,Dc: height/width/depth of the control-point grid.
    
        Assumes for simplicity that the control points lie on a regular grid.
        Can be made more general.
        """
        super(TPSGridGen3D, self).__init__()
        
        self.scale_factor = scale_factor
        self._raw_hwd = (Ho, Wo, Do)
        
        Ho = int(Ho*scale_factor)
        Wo = int(Wo*scale_factor)
        Do = int(Do*scale_factor)
        
        self._grid_hwd = (Ho, Wo, Do)        
        
        self._cp_hwd = (Hc, Wc, Dc)
    
        # The grid should be same size as the image. Except in pytorch the warping function used a normalized 
        # coordinate (-1, 1)
        # initialize the grid:
        xx, yy, zz = np.meshgrid(np.linspace(-1, 1, Ho), np.linspace(-1, 1, Wo), np.linspace(-1, 1, Do), indexing='ij')
        self._grid = np.c_[xx.flatten(), yy.flatten(), zz.flatten()].astype(np.float32)  # Nx3
        self._n_grid = self._grid.shape[0]
    
        # The number of control point should be must smaller than the grid. 
        # The control point placed regularly.
        # initialize the control points:
        xx, yy, zz = np.meshgrid(np.linspace(-1, 1, Hc), np.linspace(-1, 1, Wc), np.linspace(-1, 1, Dc), indexing='ij')
        self._control_pts = np.c_[
            xx.flatten(), yy.flatten(), zz.flatten()].astype(np.float32)  # Mx3
        self._n_cp = self._control_pts.shape[0]
    
        # compute the pair-wise distances b/w control-points and grid-points:
        Dx = ssd.cdist(self._grid, self._control_pts, metric='sqeuclidean')  # NxM
    
        # create the tps kernel:
        # real_min = 100 * np.finfo(np.float32).min
        real_min = 1e-8
        Dx = np.clip(Dx, real_min, None)  # avoid log(0)
        Kp = np.log(Dx) * Dx # the radial basis function
        Os = np.ones((self._grid.shape[0]))
        
        # c_: Translates slice objects to concatenation along the second axis.
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
        L = np.c_[Kp, np.ones((self._n_grid, 1), dtype=np.float32),
                  self._grid]  # Nx(M+4), 4 being 1, x, y, z
        self._L = torch.from_numpy(L.astype(np.float32))  # Nx(M+4), 4 being 1, x, y, z
    

    def forward(self, w_tps):
        """
        W_TPS: Bx(M+4)x3 sized tensor of tps-transformation params.
                here `M` is the number of control-points.
                    `B` is the batch-size.
    
        Returns an BxHoxWoxDox3 tensor of grid coordinates.
        """
        assert w_tps.shape[1] - 4 == self._n_cp
        batch_size = w_tps.shape[0]
        tfm_grid = torch.matmul(self._L, w_tps) # warp points
        # The parameter of the thin-plate-spline is the w_tps. The kernel L 
        # including the distance between grid/control points and the coordinate 
        # of the grid is prefixed. 
        tfm_grid = tfm_grid.flip(2).permute(0, 2, 1).reshape(
            (batch_size, 3, self._grid_hwd[0], self._grid_hwd[1], self._grid_hwd[2]))
        
        # Upsample the grid to the original size
        tfm_grid = torch.nn.functional.interpolate(tfm_grid, size=self._raw_hwd, mode='trilinear')
        tfm_grid = tfm_grid.permute(0, 2, 3, 4, 1)
        
        return tfm_grid


# This function seems to sample random w_tps to be used in the TPSGridGen.
# The thin-plate-spline contains two part: the weight for radial basis function 
# and the affine parameters.
def sample_tps_w_3D(Hc, Wc, Dc, warpsd, rotsd, scalesd, transsd):
    """
    Hc, Wc, Dc: height/width/depth of the control-point grid.
    Returns randomly sampled TPS-grid params of size (Hc*Wc*Dc+4)x3.

    Params:
      WARPSD: 2-tuple
      {ROT/SCALE/TRANS}-SD: 1-tuple of standard devs.
    """
    Nc = Hc * Wc * Dc  # no of control-pots
    # non-linear component: weight for the radial basis function
    mask = (np.random.rand(Nc, 3) > 0.5).astype(np.float32)
    W = warpsd[0] * np.random.randn(Nc, 3) + \
        warpsd[1] * (mask * np.random.randn(Nc, 3))
    
    # affine component:
    # Scaling, Rotation, Translation, 
    # The formula for rotation matrix with 3 angle is here
    # http://planning.cs.uiuc.edu/node102.html
    rnd = np.random.randn
    a, b, r = np.deg2rad(rnd() * rotsd), np.deg2rad(rnd() * rotsd), np.deg2rad(rnd() * rotsd)
    sc = 1.0 + rnd() * scalesd
    aff = [[transsd*rnd(),      transsd*rnd(),   transsd*rnd()],
           [sc * np.cos(a) * np.cos(b),   sc * (np.cos(a) * np.sin(b) * np.sin(r) - np.sin(a) * np.cos(r)),  sc * (np.cos(a) * np.sin(b) * np.cos(r) + np.sin(a) * np.sin(r))],
           [sc * np.sin(a) * np.cos(b),   sc * (np.sin(a) * np.sin(b) * np.sin(r) + np.cos(a) * np.cos(r)),  sc * (np.sin(a) * np.sin(b) * np.cos(r) - np.cos(a) * np.sin(r))],
           [sc * -np.sin(b)           ,   sc * np.cos(b) * np.sin(r)                                      ,  sc * np.cos(b) * np.cos(r)]
           ]
    W = np.r_[W, aff] # Concat the random component and affine components
    return W
