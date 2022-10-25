import random
import time
import warnings
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

import utils
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy, ImageClassifier
from tllib.modules.kernels import GaussianKernel
from tllib.modules.entropy import entropy
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

from cvxopt import matrix, solvers
import alexnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    logger = CompleteLogger(args.log, args.phase)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    train_source_dataset, \
    train_target_dataset, \
    val_dataset, \
    test_dataset, \
    num_classes, \
    args.class_names = utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)

    train_source_loader = DataLoader(train_source_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.workers,
                                     drop_last=True)

    train_target_loader = DataLoader(train_target_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.workers,
                                     drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    args.iters_per_epoch = len(train_source_iter)
    args.print_freq = args.iters_per_epoch-1

    # create model
    print("=> using model '{}'".format(args.arch))
    classifier = alexnet.AlexNet(num_classes = num_classes).to(device)


    # define optimizer
    optimizer = SGD([{'params': classifier.conv1.parameters(), 'lr':1},
                     {'params': classifier.norm1.parameters(), 'lr': 1},
                     {'params': classifier.conv2.parameters(), 'lr': 1},
                     {'params': classifier.norm2.parameters(), 'lr': 1},
                     {'params': classifier.conv345.parameters(), 'lr': 1},
                     {'params': classifier.fc6.parameters(), 'lr': 1},
                     {'params': classifier.fc7.parameters(), 'lr': 1},
                     {'params': classifier.bottleneck.parameters(), 'lr': 10},
                     {'params': classifier.classfier.parameters(), 'lr': 10}],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.wd,
                    nesterov=True)

    # Define lr scheduler
    # total_steps = args.epochs * args.iters_per_epoch
    # lr_update_rule = lambda x: args.lr / ((1 + args.lr_gamma * x/total_steps) ** args.lr_decay)
    lr_update_rule = lambda x: args.lr / ((1 + args.lr_gamma * x) ** args.lr_decay)
    lr_scheduler = LambdaLR(optimizer, lr_update_rule)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):

        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator,
          train_target_iter: ForeverDataIterator,
          model: alexnet.AlexNet,
          optimizer: SGD,
          lr_scheduler: LambdaLR,
          epoch: int,
          args):

    # Meters that used to record the training data in format: {name}: {val} ({avg})
    losses = AverageMeter('Total Loss', ':3.4f')
    Source_losses = AverageMeter('Source', ':3.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.2f')

    # Progress Meter for displaying the training information
    progress = ProgressMeter(args.iters_per_epoch,
                             [losses, Source_losses, cls_accs],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i in range(args.iters_per_epoch):

        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # compute output
        fc6_s, fc7_s, fc8_s, ys = model(x_s)

        # Compute source error
        cls_loss = F.cross_entropy(ys, labels_s)

        # The total loss function
        loss = cls_loss

        # The accuracy of 1 source batch
        cls_acc = accuracy(ys, labels_s)[0]

        # Update all the meters
        losses.update(loss.item(), x_s.size(0))
        Source_losses.update(cls_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % args.print_freq == 0:
            progress.display(i)

if __name__ == '__main__':

    class Args:
        # Dataset Parameters
        class_names = None
        root = 'running_outputs/data/office31'  # root path of dataset
        data = 'Office31'
        source = ['A']  # 2817
        target = ['W']  # 795
        train_resizing = 'default'
        val_resizing = 'default'
        resize_size = 224  # the image size after resizing
        scale = [0.08, 1.0]  # Random resize scale (default: 0.08 1.0)
        ratio = [0.75, 1.33]  # Random resize aspect ratio (default: 0.75 1.33)
        no_hflip = False  # no random horizontal flipping during training
        norm_mean = (0.485, 0.456, 0.406)  # normalization mean
        # norm_mean = (0.7085, 0.7042, 0.7029)
        norm_std = (0.229, 0.224, 0.225)  # normalization std
        # norm_std = (0.3251, 0.3263, 0.3278)

        # model parameters
        arch = 'AlexNet'
        bottleneck_dim = 256  # Dimension of bottleneck
        no_pool = False  # no pool layer after the feature extractor
        scratch = False  # whether train from scratch
        non_linear = False  # whether not use the linear version
        trade_off_lambda = 1.0  # the trade-off hyper-parameter lambda for transfer loss
        trade_off_gamma = 0.1  # the trade-off hyper-parameter gamma for entropy minimization loss

        # training parameters
        batch_size = 32
        lr = 0.001
        lr_gamma = 0.0003  # 我把这里改掉了，原值0.0003
        lr_decay = 0.75
        momentum = 0.9
        wd = 0.0005  # weight decay (default: 5e-4)
        workers = 0  # number of data loading workers (default: 2)
        epochs = 200
        iters_per_epoch = 500  # Number of iterations per epoch
        print_freq = 100  # print frequency (default: 100)
        seed = 0  # seed for initializing training
        per_class_eval = False  # whether output per-class accuracy during evaluation
        log = 'running_outputs/logs/dan/{}2{}'.format(source[0], target[0])  # Where to save logs, checkpoints and debugging images
        phase = 'train'  # When phase is 'test', only test the model; When phase is 'analysis', only analysis the model

        # My parameters
        Entropy_Minimization = False
        MKMMD_loss = False
        Maximize_MKMMD = False
        MultiLayer_Adaptation = False


    args = Args()
    main(args)