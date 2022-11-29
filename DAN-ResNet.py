import random
import time
import warnings
import shutil
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    print('''Program Started: 
             Entropy_Minimization: {}
             MKMMD_loss: {}
             Maximize_MKMMD: {}
             MultiLayer_Adaptation: {}'''.format(args.Entropy_Minimization,
                                                 args.MKMMD_loss,
                                                 args.Maximize_MKMMD,
                                                 args.MultiLayer_Adaptation))

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
    # args.iters_per_epoch = max(len(train_source_iter), len(train_target_iter))

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)


    # define optimizer
    optimizer = SGD(classifier.get_parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.wd,
                    nesterov=True)

    # Define lr scheduler
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define the MK-MMD loss function
    kernels = [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
    # kernels = [GaussianKernel(alpha=2 ** k) for k in np.linspace(-5,5,21).tolist()]
    beta_pl5 = [1 / len(kernels)] * len(kernels)
    beta_fc = [1 / len(kernels)] * len(kernels)
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(kernels=kernels, linear=not args.non_linear)

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

        # Visualize Learning Rate at the beginning of each epoch
        print('Epoch {}: Learning Rate for backbone: {}; '
              'Learning Rates for bottleneck: {}; '
              'Learning Rates for head: {}'.format(epoch,
                                                   optimizer.state_dict()['param_groups'][0]['lr'],
                                                   optimizer.state_dict()['param_groups'][1]['lr'],
                                                   optimizer.state_dict()['param_groups'][2]['lr']))

        # train for one epoch
        beta_pl5, beta_fc = train(train_source_iter,
                                  train_target_iter,
                                  classifier,
                                  mkmmd_loss,
                                  optimizer,
                                  lr_scheduler,
                                  epoch,
                                  beta_pl5,
                                  beta_fc,
                                  kernels,
                                  args)

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
          model: ImageClassifier,
          mkmmd_loss: MultipleKernelMaximumMeanDiscrepancy,
          optimizer: SGD,
          lr_scheduler: LambdaLR,
          epoch: int,
          beta_pl5: list,
          beta_fc: list,
          kernels: list,
          args):

    # Meters that used to record the training data in format: {name}: {val} ({avg})
    losses = AverageMeter('Total Loss', ':3.4f')
    Source_losses = AverageMeter('Source', ':3.4f')
    trans_losses = AverageMeter('MMD', ':5.4f')
    entropy_losses = AverageMeter('Entropy', ':5.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.2f')

    # Progress Meter for displaying the training information
    progress = ProgressMeter(args.iters_per_epoch,
                             [losses, Source_losses, entropy_losses, trans_losses, cls_accs],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    mkmmd_loss.train()

    # If you wish to maximize MK-MMD after each epoch (update β), then we need to create the following variables
    if args.Maximize_MKMMD:
        if args.MultiLayer_Adaptation:
            covar_matrix_pl5 = torch.zeros((len(beta_pl5), len(beta_pl5)))
            covar_matrix_fc = torch.zeros((len(beta_fc), len(beta_fc)))
            M_k_pl5 = torch.zeros((len(beta_pl5), ))
            M_k_fc = torch.zeros((len(beta_fc), ))
        else:
            covar_matrix_fc = torch.zeros((len(beta_fc), len(beta_fc)))
            M_k_fc = torch.zeros((len(beta_fc),))

    for i in range(args.iters_per_epoch):

        # Record the training process (p will be used for scaling the trade-off parameters of Entropy Minimization and Mk-MMD loss)
        p = (i + args.iters_per_epoch * epoch) / (10 * args.iters_per_epoch)

        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        pl5_s, fc_s, y_s = model(x_s)
        pl5_t, fc_t, y_t = model(x_t)

        # Compute source error
        cls_loss = F.cross_entropy(y_s, labels_s)

        # Compute Entropy Minimization loss here
        if args.Entropy_Minimization:
            entropy_min_loss = entropy(F.softmax(y_t), reduction='mean')
        else:
            entropy_min_loss = 0

        # Compute MK-MMD loss here
        if args.MKMMD_loss:
            if args.MultiLayer_Adaptation:
                transfer_loss = mkmmd_loss(pl5_s, pl5_t, beta_pl5) + mkmmd_loss(fc_s, fc_t, beta_fc)
            else:
                transfer_loss = mkmmd_loss(fc_s, fc_t, beta_fc)
            # transfer_loss = torch.max(transfer_loss, torch.scalar_tensor(0))
            # transfer_loss = torch.abs(transfer_loss)
        else:
            transfer_loss = 0

        # The total loss function
        gamma = args.trade_off_gamma * (2 / (1 + np.e ** (-10 * p)) - 1)
        lbd = args.trade_off_lambda * (2 / (1 + np.e ** (-10 * p)) - 1)
        loss = cls_loss + (entropy_min_loss * gamma) + (transfer_loss * lbd)

        # The accuracy of 1 source batch
        cls_acc = accuracy(y_s, labels_s)[0]

        # Update all the meters
        losses.update(loss.item(), x_s.size(0))
        Source_losses.update(cls_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        if args.MKMMD_loss:
            trans_losses.update(transfer_loss.item(), x_s.size(0))
        if args.Entropy_Minimization:
            entropy_losses.update(entropy_min_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Compute the Quu' values w.r.t each fc layer
        if args.Maximize_MKMMD:
            if args.MultiLayer_Adaptation:
                covar_matrix_pl5, M_k_pl5 = Update_covariance_matrix_one_epoch(pl5_s, pl5_t, kernels, covar_matrix_pl5, i, M_k_pl5)
                covar_matrix_fc, M_k_fc = Update_covariance_matrix_one_epoch(fc_s, fc_t, kernels, covar_matrix_fc, i, M_k_fc)
            else:
                covar_matrix_fc, M_k_fc = Update_covariance_matrix_one_epoch(fc_s, fc_t, kernels, covar_matrix_fc, i, M_k_fc)

        if i % args.print_freq == 0:
            progress.display(i)

    # After an entire epoch is trained, Update β
    if args.Maximize_MKMMD:
        if args.MultiLayer_Adaptation:
            print('All the covariance matrices recorded, start calculating BETAs...')
            beta_pl5 = Calculate_beta_based_on_covariance_matrix(covar_matrix_pl5, M_k_pl5)
            beta_fc = Calculate_beta_based_on_covariance_matrix(covar_matrix_fc, M_k_fc)
            print('beta_pl5 updated to: {}'.format(beta_pl5))
            print('beta_fc updated to: {}'.format(beta_fc))
        else:
            print('All the covariance matrices recorded, start calculating BETAs...')
            beta_fc = Calculate_beta_based_on_covariance_matrix(covar_matrix_fc, M_k_fc)
            print('beta_fc updated to: {}'.format(beta_fc))

    return beta_pl5, beta_fc

def Update_covariance_matrix_one_epoch(f_s: torch.Tensor,
                                       f_t: torch.Tensor,
                                       kernels: list,
                                       covar_matrix: torch.Tensor,
                                       count: int,
                                       M_k: torch.Tensor):

    num_kernels = len(kernels)

    with torch.no_grad():

        features = torch.cat([f_s, f_t], dim=0)

        # Iterate through the kernels
        for i in range(num_kernels):

            B_u = [1]
            k_u = [kernels[i]]
            MkMMD_loss = MultipleKernelMaximumMeanDiscrepancy(kernels=k_u, linear=not args.non_linear)
            M_u = 0

            kernel1 = kernels[i]
            kernel1_matrix = kernel1(features)
            k1_ss = kernel1_matrix[0:args.batch_size, 0:args.batch_size]
            k1_tt = kernel1_matrix[args.batch_size:, args.batch_size:]
            k1_st = kernel1_matrix[0:args.batch_size, args.batch_size:]

            for j in range(i, num_kernels):

                kernel2 = kernels[j]
                kernel2_matrix = kernel2(features)
                k2_ss = kernel2_matrix[0:args.batch_size, 0:args.batch_size]
                k2_tt = kernel2_matrix[args.batch_size:, args.batch_size:]
                k2_st = kernel2_matrix[0:args.batch_size, args.batch_size:]


                if j == i:
                    M_u += MkMMD_loss(f_s, f_t, B_u)

                # Linear time estimation Quu'
                idx = 0
                Quu = []
                while idx < len(f_s) // 4:    # 0,1,2,3,4,5,6,7
                    t1 = (k1_ss[4 * idx, 4 * idx + 1] +
                          k1_tt[4 * idx, 4 * idx + 1] -
                          k1_st[4 * idx, 4 * idx + 1] -
                          k1_st[4 * idx + 1, 4 * idx])
                    t2 = (k1_ss[4 * idx + 2, 4 * idx + 3] +
                          k1_tt[4 * idx + 2, 4 * idx + 3] -
                          k1_st[4 * idx + 2, 4 * idx + 3] -
                          k1_st[4 * idx + 3, 4 * idx + 2])
                    t3 = (k2_ss[4 * idx, 4 * idx + 1] +
                          k2_tt[4 * idx, 4 * idx + 1] -
                          k2_st[4 * idx, 4 * idx + 1] -
                          k2_st[4 * idx + 1, 4 * idx])
                    t4 = (k2_ss[4 * idx + 2, 4 * idx + 3] +
                          k2_tt[4 * idx + 2, 4 * idx + 3] -
                          k2_st[4 * idx + 2, 4 * idx + 3] -
                          k2_st[4 * idx + 3, 4 * idx + 2])

                    Quu.append((t1 - t2) * (t3 - t4))
                    idx += 1

                # Put Quu' at the correct position of Q
                res = sum(Quu) / len(Quu)
                covar_matrix[i][j] = covar_matrix[i][j] * count / (count + 1) + res / (count + 1)
                covar_matrix[j][i] = covar_matrix[i][j]

            M_k[i] = M_k[i] * count / (count + 1) + M_u / (count + 1)

    return covar_matrix, M_k

def Calculate_beta_based_on_covariance_matrix(covar_matrix: torch.Tensor, M_k: torch.Tensor):

    num_kernels = covar_matrix.shape[0]

    epsilon = .001
    reg = torch.eye(num_kernels, requires_grad=False) * epsilon  # εI
    covar_matrix = covar_matrix + reg  # Q + εI

    # 使用cvxopt包求解二次规划问题（from cvxopt import matrix, solvers）
    P = covar_matrix.t().numpy().tolist()  # (Q + εI)'
    q = [0. for i in range(num_kernels)]
    G = (torch.eye(num_kernels, requires_grad=False) * -1).t().numpy().tolist()
    h = [0. for i in range(num_kernels)]    # 0
    # A = [[1.] for i in range(num_kernels)]    # 这个A就是原式中的M，原式中的M中每个值都是对应的kernel计算出的MMD loss
    A = [[x] for x in M_k.numpy().tolist()]    # M_k
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix([1.0])
    solvers.options['show_progress'] = False
    result = solvers.qp(P, q, G, h, A, b)

    return list(result['x'])

if __name__ == '__main__':

    class Args:
        # Dataset Parameters
        class_names = None  # Set it to None for now, it will be automatically updated before training
        root = 'running_outputs/data/office31'  # root path of dataset
        data = 'Office31'  # On which dataset do you want to try out your model
        source = ['A']  # Amazon: 2817 images; Webcam: 795 images; DSLR: 498 images
        target = ['W']
        train_resizing = 'default'  # Don't worry and leave this as 'default'
        val_resizing = 'default'
        resize_size = 224  # the image size after resizing
        scale = [0.08, 1.0]  # Random resize scale (default: 0.08 1.0)
        ratio = [0.75, 1.33]  # Random resize aspect ratio (default: 0.75 1.33)
        no_hflip = False  # no random horizontal flipping during training
        norm_mean = (0.485, 0.456, 0.406)  # normalization mean
        norm_std = (0.229, 0.224, 0.225)  # normalization std

        # model parameters
        arch = 'resnet18'
        bottleneck_dim = 256  # Dimension of bottleneck
        no_pool = False  # Whether not to use pooling layer after the feature extractor
        scratch = False  # whether not to train from scratch
        non_linear = True  # Whether not use the linear version
        trade_off_lambda = 1.0  # the trade-off hyper-parameter lambda for transfer loss
        trade_off_gamma = 0.1  # the trade-off hyper-parameter gamma for entropy minimization loss

        # training parameters
        batch_size = 32
        lr = 0.003
        lr_gamma = 0.0003  # 我把这里改掉了，原值0.0003
        lr_decay = 0.75
        momentum = 0.9
        wd = 0.0005  # weight decay (default: 5e-4)
        workers = 0  # number of data loading workers (default: 2)
        epochs = 20
        iters_per_epoch = 88  # Number of iterations per epoch
        print_freq = 87  # print frequency (default: 100)
        seed = 0  # seed for initializing training
        per_class_eval = False  # whether output per-class accuracy during evaluation
        log = 'running_outputs/logs/dan/{}2{}'.format(source[0], target[0])  # Where to save logs, checkpoints and debugging images
        phase = 'train'  # When phase is 'test', only test the model; When phase is 'analysis', only analysis the model

        # My parameters
        Entropy_Minimization = True  # Whether to include Entropy Minimization in the training procedure
        MKMMD_loss = True  # Whether to include MK-MMD loss in the training procedure
        Maximize_MKMMD = True  # Whether to Update β after one epoch of training (update β w.r.t maximizing MK-MMD)
        MultiLayer_Adaptation = True  # Whether to adapt multiple layers (include MK-MMD values calculated from multiple layers in the training procedure)

    args = Args()
    main(args)