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
    logger = get_logger('RDA' + model_name + 'train', log_file=log_name + '/train.log')
    logger.info(f'Start training on {device}.')

    # training loops
    best_val_score = 1000000000

    
    for epoch in range(args.initial_epoch, args.epochs):

        epoch_step_time = []

        utils.train(train_loader, model, model_dir, 
          losses, 
          optimizer, 
          epoch, 
          writer, 
          tensorboard_formatter, 
          args.log_after_iters, 
          logger, 
          device, 
          risi_loss_weight = args.risi_loss_weight, 
          save_image = args.save_image, 
          save_image_freq = args.save_image_freq, 
          epochs = args.epochs,)


 
        ########################################################################
        
        # save checkpoint
        if epoch % args.save_model_per_epochs == 0:
            last_file_path = os.path.join(model_dir, 'last_checkpoint_%04d.pt' % epoch)
            attention_file_path = os.path.join(model_dir, 'attention_last_checkpoint_%04d.pt' % epoch)
            model.save(last_file_path)
            # model.attention_model.save(attention_file_path)

        ########################################################################

        # validation
        if epoch % args.validate_after_epochs == 0:

            logger.info(f'Start performing validation.')

            val_epoch_total_loss = utils.validate(eval_loader, model, 
                losses, 
                optimizer, 
                epoch, 
                writer, 
                logger, 
                device, 
                len(train_loader), 
                risi_loss_weight = args.risi_loss_weight, 
                epochs = args.epochs,)

        # save checkpoint
        if val_epoch_total_loss.avg < best_val_score:
            best_file_path = os.path.join(model_dir, 'best_checkpoint_%04d.pt' % epoch)
            shutil.copyfile(last_file_path, best_file_path)
            best_val_score = val_epoch_total_loss.avg


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
        parse.add_argument('--save-image-freq', default=300,
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