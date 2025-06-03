#!/usr/bin/env python

"""
Script to train a Regional Deep Atrophy model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

Registration will be intra-subject scan-to-scan.
"""


# from comet_ml import Experiment
import os
import argparse
import time
import shutil
import numpy as np
import importlib
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from functools import partial
import pickle
from . import utils
from . import generators
from .utils import get_logger, get_tensorboard_formatter, RunningAverage
from . import layers
from . import networks
import matplotlib
from datetime import datetime
import torch.multiprocessing as mp

matplotlib.use('Agg')  # MUST BE CALLED BEFORE IMPORTING plt, or qt5agg for interactive plotting, Agg for non-interactive, such as saving to file

def _create_optimizer(model, lr, weight_decay):
    learning_rate = lr
    weight_decay = weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _create_lr_scheduler(optimizer, lr_scheduler=None):
    lr_config = {"name": lr_scheduler,
                 "milestones": [30, 40, 50],
                 "gamma": 0.2 }

    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)


def main(args):

    print("ROOT is ", args.ROOT)

    if not os.path.exists(args.ROOT + '/Model'):
        os.makedirs(args.ROOT + '/Model')

    if not os.path.exists(args.ROOT + '/log'):
        os.makedirs(args.ROOT + '/log')

    # extract shape from sampled input
    # inshape = (48, 80, 64)
    inshape = (args.input_D, args.input_H, args.input_W)
    print("inshape is", inshape)

    train_augment = ['normalize', 'rotate', 'random_crop'] # 'erase' , 'rotate', 'flip'
    # in training, flip is not needed because they are automatically included in rotation.
    eval_augment = ['normalize', 'fixed_crop']

    train_dataset = generators.DatasetRISI3D(
                                args.train_double_pairs,
                                train_augment,
                                add_feat_axis=args.add_feat_axis,
                                output_size=inshape,
                                risi_categories=args.risi_categories,
                                max_angle=args.max_angle,
                                rotate_prob=args.rotate_prob,)
    
    
    eval_dataset = generators.DatasetRISI3D(
                                args.eval_double_pairs,
                                eval_augment,
                                add_feat_axis=args.add_feat_axis,
                                output_size=inshape,
                                risi_categories=args.risi_categories,
                                max_angle=args.max_angle,
                                rotate_prob=args.rotate_prob,)


    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            # sampler = train_sampler,
            num_workers=args.workers)
            # persistent_workers=True)
        
    eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers)

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    # torch.cuda.set_device(device)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    print("torch.backends.cudnn.deterministic", torch.backends.cudnn.deterministic)

    print("load model")

    if args.load_model:
        # load the whole model (if specified)
        model = networks.SingleAttentionALOHA.load(args.load_model, device)

    else:
        if args.load_unet:
            UNET_model = networks.UNet3D_Seg.load(args.load_unet, device)
        else:
            UNET_model = networks.UNet3D_Seg(in_dim=args.in_dim, 
                                             n_classes = args.n_classes, 
                                             out_dim=args.out_dim)
        
        model = networks.SingleAttentionALOHA(inshape=inshape,
                                            unet_model=UNET_model,
                                            hyper_a=args.hyper_a_sigmoid,
                                            risi_categories = args.risi_categories)
        

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    model.to(device)
    model.train()

    optimizer = _create_optimizer(model, args.lr, args.weight_decay)
    lr_scheduler = _create_lr_scheduler(optimizer, args.lr_scheduler)

    if args.sto_loss == 'ce':
        sto_loss_func = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        raise ValueError('STO loss should be "ce", but found "%s"' % args.sto_loss)

    if args.risi_loss == 'ce':
        risi_loss_func = torch.nn.CrossEntropyLoss().cuda(args.gpu)
        # radius = [args.radius] * len(inshape)
        # image_loss_func = losses.NCC(radius).loss # best practice is radius = 2 or window size = 5, in greedy # win=5
    elif args.risi_loss == 'mse':
        risi_loss_func = torch.nn.MSELoss()
    else:
        raise ValueError('RISI loss should be "ce" or "mse", but found "%s"' % args.risi_loss)

    losses = [sto_loss_func, sto_loss_func, sto_loss_func, sto_loss_func, risi_loss_func, risi_loss_func]

    print("losses: ", losses)

    # prepare deformation loss
    # regularization loss is always l2
    # for STO training, no regularization loss
    # losses += [losses.Grad('l2', loss_mult=args.int_downsize).loss]
    # weights += [args.weight]

    curr_time = time.strftime("%Y-%m-%d_%H-%M")
    print("model_name:", curr_time)
    model_name = curr_time + 'train_'

    model_dir = args.ROOT + "/Model/" + curr_time
    os.makedirs(model_dir, exist_ok=True)

    log_name = (args.ROOT + "/log/" + curr_time)
    writer = SummaryWriter(log_name)
    tensorboard_formatter = get_tensorboard_formatter()

    # create logger
    logger = get_logger('RDA' + model_name + 'train')
    logger.info(f'Start training on {device}.')


    # initialize log variables
    train_epoch_loss = []
    val_epoch_loss = []

    train_epoch_total_acc = []
    val_epoch_total_acc = []

    for n in range(len(losses)):
        train_epoch_loss.append(RunningAverage())
        val_epoch_loss.append(RunningAverage())

        train_epoch_total_acc.append(RunningAverage())
        val_epoch_total_acc.append(RunningAverage())

    train_epoch_total_loss = RunningAverage()
    val_epoch_total_loss = RunningAverage()

    # training loops
    best_val_score = 1000000000

    
    for epoch in range(args.initial_epoch, args.epochs):

        epoch_step_time = []

        # reset log variables
        for n in range(len(losses)):
            train_epoch_loss[n].reset()
            train_epoch_total_acc[n].reset()
        train_epoch_total_loss.reset()

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

                    train_epoch_loss[n].update(curr_loss.item(), args.batch_size)
                    train_epoch_total_acc[n].update(train_acc.item(), args.batch_size)
                    loss += curr_loss

                elif n < 4:
                    # loss 1: ce for scan temporal order for the second pair
                    curr_loss = loss_function(vol_diff_label[n], sample["label_date_diff2"].long())
                    date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                    train_acc = torch.mean((date_diff_pred == sample["label_date_diff2"]).double())

                    train_epoch_loss[n].update(curr_loss.item(), args.batch_size)
                    train_epoch_total_acc[n].update(train_acc.item(), args.batch_size)
                    loss += curr_loss

                else:
                    # loss 2: mse for RISI
                    curr_loss = loss_function(vol_diff_label[n], sample["label_time_interval"].long())
                    train_epoch_loss[n].update(curr_loss.item(), args.batch_size)

                    label_time_interval_pred = torch.argmax(vol_diff_label[n], dim=1)
                    train_acc = torch.mean((label_time_interval_pred == sample["label_time_interval"]).double())
                    train_epoch_total_acc[n].update(train_acc.item(), args.batch_size)

                    loss += args.risi_loss_weight * curr_loss


            train_epoch_total_loss.update(loss.item(), args.batch_size)

            optimizer.zero_grad()
            loss.backward()
            # print(torch.autograd.grad(loss, vol_diff_label[5]))
            optimizer.step()
            # model.UNet3D_Seg.decoder1.conv_block.conv1.conv.weight

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            # log training information
            iterations = len(train_loader) * epoch + i
            if iterations % args.log_after_iters == 0:
                loss_text = ""
                for n in range(len(losses)):
                    loss_text = loss_text + f'{train_epoch_loss[n].avg},'
                logger.info(
                    f'Training stats. Epoch: {epoch + 1}/{args.epochs}. ' +
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
                # log_images(writer, inputs, filename, y_pred, (fixed[0] - y_pred[0] + 1) * 0.5, iterations)
                # parameters: writer, inputs, prediction, diff, num_iterations

            if args.save_image and iterations % args.save_image_freq == 0: # save_image_freq = 100
                # save a batch of (10) images at a time
                for i in range(len(sample["label_date_diff1"])//2):

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
                    utils.save_volfile(sample["imgs_bl1"][i, 0, :].detach().cpu().numpy().squeeze(), moving_name1)
                    utils.save_volfile(sample["imgs_bl2"][i, 0, :].detach().cpu().numpy().squeeze(), moving_name2)

                    # save fixed image
                    utils.save_volfile(sample["imgs_fu1"][i, 0, :].detach().cpu().numpy().squeeze(), fixed_name1)
                    utils.save_volfile(sample["imgs_fu2"][i, 0, :].detach().cpu().numpy().squeeze(), fixed_name2)

                    # save attention image
                    moving_attention_name1 = moving_name1.replace('.nii.gz', "_attention.nii.gz", 1)
                    utils.save_volfile(attention_pred1[i, :].detach().cpu().numpy().squeeze().astype(float), moving_attention_name1)

                    moving_attention_name2 = moving_name2.replace('.nii.gz', "_attention.nii.gz", 1)
                    utils.save_volfile(attention_pred2[i, :].detach().cpu().numpy().squeeze().astype(float), moving_attention_name2)

                    # # save jacobian determinant
                    # moving_jdet_name1 = moving_name1.replace('.nii.gz', "_jdet.nii.gz", 1)
                    # utils.save_volfile(jdet1[i, 0, :].detach().cpu().numpy().squeeze(), moving_jdet_name1)
                    
                    # moving_jdet_name2 = moving_name2.replace('.nii.gz', "_jdet.nii.gz", 1)
                    # utils.save_volfile(jdet1[i, 0, :].detach().cpu().numpy().squeeze(), moving_jdet_name2)

                    # save seg image before registration
                    utils.save_volfile(sample["imgs_seg1"][i, 0, :].detach().cpu().numpy().squeeze(), moving_seg_name1)
                    utils.save_volfile(sample["imgs_seg2"][i, 0, :].detach().cpu().numpy().squeeze(), moving_seg_name2)

                    # save moved image
                    utils.save_volfile(moved_img1[i, 0, :].detach().cpu().numpy().squeeze(), moved_name1)
                    utils.save_volfile(moved_img2[i, 0, :].detach().cpu().numpy().squeeze(), moved_name2)

                    # save warp image
                    utils.save_volfile(sample["imgs_warp1"][i, :].detach().cpu().permute(1, 2, 3, 0).numpy().squeeze(), warp_name1)
                    utils.save_volfile(sample["imgs_warp2"][i, :].detach().cpu().permute(1, 2, 3, 0).numpy().squeeze(), warp_name2)

        ########################################################################
        
        # save checkpoint
        if epoch % args.save_model_per_epochs == 0:
            last_file_path = os.path.join(model_dir, 'last_checkpoint_%04d.pt' % epoch)
            attention_file_path = os.path.join(model_dir, 'attention_last_checkpoint_%04d.pt' % epoch)
            model.save(last_file_path)
            # model.attention_model.save(attention_file_path)
            if val_epoch_total_loss.avg < best_val_score:
                best_file_path = os.path.join(model_dir, 'best_checkpoint.pt')
                shutil.copyfile(last_file_path, best_file_path)
                best_val_score = val_epoch_total_loss.avg

        ########################################################################

        # validation
        if epoch % args.validate_after_epochs == 0:

            logger.info(f'Start performing validation.')

            # reset log variables
            for n in range(len(losses)):
                val_epoch_loss[n].reset()
                val_epoch_total_acc[n].reset()
            val_epoch_total_loss.reset()

            # perform validation
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

                            val_epoch_loss[n].update(curr_loss_val.item(), args.batch_size)
                            val_epoch_total_acc[n].update(val_acc.item(), args.batch_size)

                        elif n < 4:
                            # loss 1: ce for scan temporal order for the second pair
                            curr_loss_val = loss_function(vol_diff_label[n], sample[
                                "label_date_diff2"].long())  # * args.train_batch_size / math.prod(inshape)
                            date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                            val_acc = torch.mean((date_diff_pred == sample["label_date_diff2"]).double())

                            val_epoch_loss[n].update(curr_loss_val.item(), args.batch_size)
                            val_epoch_total_acc[n].update(val_acc.item(), args.batch_size)

                        else:
                            # loss 2: mse for RISI
                            curr_loss_val = loss_function(vol_diff_label[n], sample[
                                "label_time_interval"].long())  # * args.train_batch_size / math.prod(inshape)
                            val_epoch_loss[n].update(curr_loss_val.item(), args.batch_size)

                            label_time_interval_pred = torch.argmax(vol_diff_label[n], dim=1)
                            val_acc = torch.mean((label_time_interval_pred == sample["label_time_interval"]).double())
                            val_epoch_total_acc[n].update(val_acc.item(), args.batch_size)

                        loss_val += curr_loss_val

            val_epoch_total_loss.update(loss_val.item(), args.batch_size)

            # log stats
            loss_text = ""
            for n in range(len(losses)):
                loss_text = loss_text + f'{val_epoch_loss[n].avg},'
            logger.info(
                f'Validation stats. Epoch: {epoch + 1}/{args.epochs}. ' +
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
                    len(train_loader) * (epoch + 1))
            log_epoch_lr(writer, epoch, optimizer, len(train_loader) * (epoch + 1))



            # adjust learning rate if necessary after each epoch
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(val_epoch_total_loss.avg)
        else:
            lr_scheduler.step()

        # print epoch info
        # epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        # time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        # losses_info = ', '.join(['%.4e' % f.avg for f in train_epoch_loss])
        # loss_info = 'loss: %.4e  (%s)' % (train_epoch_total_loss.avg, losses_info)
        # print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    # final model save
    model.save(os.path.join(model_dir, model_name + '%04d.pt' % args.epochs))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

class RDATrainLauncher:

    def __init__(self, parse):
        ############################# data organization parameters ############################
        parse.add_argument('--gpu', default='0', 
                           help='GPU ID number(s), comma-separated (default: 0)')
        parse.add_argument('--train-double-pairs', default=None, 
                            help='line-seperated list of training files')
        parse.add_argument('--eval-double-pairs', default=None, 
                            help='line-seperated list of validation files')
        parse.add_argument('--img-prefix', help='optional input image file prefix')
        parse.add_argument('--img-suffix', help='optional input image file suffix')
        parse.add_argument('--ROOT', metavar='DIR',
                            default="/data/mengjin/RegionalDeepAtrophy",
                            help='directory to save models and logs')
        parse.add_argument('--add-feat-axis', default=False,
                            help='when loading with gpu, do not need to add feature axis.')
        parse.add_argument('--save-image', default=True,
                            help='Whether to save tested images to .nii.gz file formats')
        parse.add_argument('--save-image-freq', default=200,
                            help='Whether to save tested images to .nii.gz file formats')

        ############################# training parameters ############################
        parse.add_argument('--n-classes', type=int, default=2, 
                            help='number of output classes for unet segmentation')
                            # default = 2: [nothing, shrink, expand]
        parse.add_argument('--in-dim', type=int, default=2, 
                            help='default dimension for unet seg input')
                            # 2: input 2 images for [bl, fu] for two attention maps
        parse.add_argument('--out-dim', type=int, default=3, 
                            help='default dimension for unet seg output')
                            # 1: output attention map directly,
                            # 2: output 2 channels mimicking segmentation images
                            # 3: output 3 channels of [shrink, bg, expand] voxels
        parse.add_argument('--input-D',default=48,type=int,
                           help='Input size of depth')
        parse.add_argument('--input-H', default=80, type=int,
                           help='Input size of height')
        parse.add_argument('--input-W', default=64, type=int,
                           help='Input size of width')
        parse.add_argument('--hyper-a-sigmoid', type=float, default=0.02, 
                            help='default hyper parameter for sigmoid for RISI loss')
        parse.add_argument('--risi-categories', type=int, default=4, 
                            help='default number of label categories for RISI loss')
        parse.add_argument('--risi-loss-weight', type=float, default=1, 
                            help='default weight of risi loss')
        parse.add_argument('--epochs', type=int, default=100,
                            help='number of training epochs (default: 100)')
        parse.add_argument('--log-after-iters', type=int, default=20, help='')
        parse.add_argument('--validate-after-epochs', type=int, default=1, 
                           help='validate after # epochs')
        parse.add_argument('--save-model-per-epochs', type=int, default=1, 
                           help='save mocel after each # epochs')
        parse.add_argument('--load-model',
                            default="",
                            # default="/voxelmorph_20210907/Model/2023-08-20_20-30/best_checkpoint.pt", # ncc = 5.33, /last_checkpoint_0030.pt
                            help='optional model file to initialize with')
        parse.add_argument('--load-unet',
                            default="",
                            help='optional model file to initialize with')
        parse.add_argument('--initial-epoch', type=int, default=0,
                            help='initial epoch number (default: 0)')
        parse.add_argument('--lr', type=float, default=1e-4, 
                            help='learning rate (default: 1e-4)') # was 5e-4
        parse.add_argument('--weight-decay', type=float, default=1e-4, 
                            help='weight decay for learning rate (default: 1e-4)') # was 1e-4
        parse.add_argument('--lr-scheduler', type=str, default="MultiStepLR", 
                            help='learning rate scheduler') # was 1e-4
        parse.add_argument('--milestones', default=[30, 40, 50], 
                            help='milestones for learning rate scheduler')
        parse.add_argument('--gamma', type=float, default=0.2, 
                            help='gamma for learning rate scheduler')
        parse.add_argument('--cudnn-nondet', action='store_false', # original was store_true
                            help='disable cudnn determinism - might slow down training')
        parse.add_argument('--radius', type=int, default=5, help='Radius for NCC loss')
        parse.add_argument('--batch-size', type=int, default=128, 
                            help='batch size (default: 128)')
        parse.add_argument('-j', '--workers', default=0, type=int, metavar='N', # 12
                            help='number of data loading workers (default: 12)')
                            # num_workers = min(4 * num_gpus, os.cpu_count() // num_gpus)
        parse.add_argument('--max-angle', type=int, default=45,
                            help='maximum rotation angle (default: 45)')
        parse.add_argument('--rotate-prob', type=float, default=0.5,
                            help='probability of rotation (default: 0.5)')

        ############################ loss hyperparameters ############################
        parse.add_argument('--risi-loss', default='ce',
                            help='risi loss - for relative interscan interval (default: ce)')
        parse.add_argument('--sto-loss', default='ce',
                            help='sto loss - for scan temporal order (default: ce)')
        parse.add_argument('--lambda', type=float, dest='weight', default=0.1,  # 0.01 previous experiments; 0.85 for intrasubject from hypermorph; 0.15 for intersubject.
                            help='weight of deformation loss (default: 0.01)')
                            # weight of the regularization term,
                            # 1 for NCC loss
                            # 0.01 for MSE loss
                            # weights: [1, 0.01] Sunday; [1, 0.85] Monday; [1, 5.33] Wednesday.

        # Set the function to run
        # parse.set_defaults(func = lambda args : self.run(args))
        parse.set_defaults(func = lambda args : self.run(args))

    def run(self, args):

        main(args)