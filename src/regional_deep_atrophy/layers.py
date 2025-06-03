import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib



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
    T_pt = T_pt.to('cpu').numpy()

    return T_pt, img.header


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        self.size = size

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode="border")


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.inshape = inshape
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
