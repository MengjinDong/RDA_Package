from tensorboardX import SummaryWriter

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


def log_images(writer, filename, inputs, out_seg, moved_img, warp, jdet, num_iterations, segs_gt=None):
    # for images, log images to tensorboardX
    nums_display = 4 # was 10
    inputs_map = {
        'moving':          inputs[0][0:nums_display, 0, :], # input bl and fu at the same time
        'fixed':           inputs[1][0:nums_display, 0, :],
        'seg_bg':          out_seg[0:nums_display, 0, :],
        'seg_hippo':       out_seg[0:nums_display, 1, :],
        'seg_sulcus':      out_seg[0:nums_display, 2, :],
        'jdet':            jdet[0:nums_display, 0, :],
        'moved':           moved_img[0:nums_display, 0, :],
        'warp0':           warp[0:nums_display, 0, :],
        'warp1':           warp[0:nums_display, 1, :],
        'warp2':           warp[0:nums_display, 2, :],
        'difference':      (inputs[1][0:nums_display, 0, :] - moved_img[0:nums_display, 0, :] + 1) * 0.5
    }
    if segs_gt:
        inputs_map['segs_gt'] = segs_gt[0][0:nums_display, 0, :]
    image_names = filename[0:nums_display]
    img_sources = {}
    for name, batch in inputs_map.items():
        if isinstance(batch, list) or isinstance(batch, tuple):
            # if isinstance(batch[0], str):
            #     img_sources[name] = batch
            # else:
            for i, b in enumerate(batch):
                img_sources[f'{name}{i}'] = b.data.cpu().numpy()
        else:
            img_sources[name] = batch.data.cpu().numpy()
    for name, batch in img_sources.items():
        # TODO: https://stackoverflow.com/questions/60907358/how-do-i-add-an-image-title-to-tensorboardx
        # no direct way to add image title to tensorboard
        for tag, image in tensorboard_formatter(name, batch, normalize=True): # image_names
            writer.add_image(tag, image, num_iterations, dataformats='CHW')


def log_epoch_lr(writer, epoch, optimizer, num_iterations):
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('z_learning_rate', lr, num_iterations)
    writer.add_scalar('z_epoch', epoch, num_iterations)


def log_stats(writer, phase, loss, loss_list, acc1, acc2, num_iterations):
    tag_value = {
        f'{phase}_loss_avg': loss.avg,
        f'{phase}_acc1_avg': acc1.avg,
        f'{phase}_acc2_avg': acc2.avg,
    }
    for n, subloss in enumerate(loss_list):
        tag_value[f'{phase}_subloss_{n}'] = subloss.avg

    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, num_iterations)

log_name = (args.vxm_dir + "/" + args.log_dir
            + "/" + args.data_dir
            + "/" + curr_time) # define your own model name here
writer = SummaryWriter(log_name)
tensorboard_formatter = get_tensorboard_formatter()


for epoch in epochs:

    # run training and evaluation here, and log when necessary
    log_stats(writer, 'train', train_epoch_total_loss, train_epoch_loss, train_epoch_total_acc[0], train_epoch_total_acc[1],
              iterations)

    log_images(writer, filename, inputs, out_seg, moved_img, warp, jdet, iterations, segs_gt)


    log_stats(writer, 'val', val_epoch_total_loss, val_epoch_loss, val_epoch_total_acc[0], val_epoch_total_acc[1],
              args.steps_per_train_epoch * (epoch + 1))
    log_epoch_lr(writer, epoch, optimizer, args.steps_per_train_epoch * (epoch + 1))


writer.export_scalars_to_json("./all_scalars.json")
writer.close()





