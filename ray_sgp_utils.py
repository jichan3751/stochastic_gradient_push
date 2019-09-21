# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Gossip SGD

Distributed data parallel training of a ResNet-50 on ImageNet using either
- Stochastic Gradient Push
- AllReduce SGD (aka Parallel Stochastic Gradient Descent)
- Distributed Parallel Stochastic Gradient Descent

Derived from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import argparse
import copy
import os
import socket
import time
import logging
import sys


import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# from apex import amp
# from apex.parallel import DistributedDataParallel as ApexDDP
# from apex.fp16_utils import network_to_half, FP16_Optimizer
from torchvision.models.resnet import Bottleneck
from torch.nn.parameter import Parameter


from experiment_utils import make_logger
from experiment_utils import Meter
from experiment_utils import ClusterManager
from experiment_utils import get_tcp_interface_name
from gossip_module import GossipDataParallel
from gossip_module import DynamicBipartiteExponentialGraph as DBEGraph
from gossip_module import DynamicBipartiteLinearGraph as DBLGraph
from gossip_module import DynamicDirectedExponentialGraph as DDEGraph
from gossip_module import DynamicDirectedLinearGraph as DDLGraph
from gossip_module import NPeerDynamicDirectedExponentialGraph as NPDDEGraph
from gossip_module import RingGraph
from gossip_module import UniformMixing

GRAPH_TOPOLOGIES = {
    0: DDEGraph,    # Dynamic Directed Exponential
    1: DBEGraph,    # Dynamic Bipartite Exponential
    2: DDLGraph,    # Dynamic Directed Linear
    3: DBLGraph,    # Dynamic Bipartite Linear
    4: RingGraph,   # Ring
    5: NPDDEGraph,  # N-Peer Dynamic Directed Exponential
    -1: None,
}

MIXING_STRATEGIES = {
    0: UniformMixing,  # assign weights uniformly
    -1: None,
}



def get_optimizer_creator(model, config):

    def criterion(input, kl_target):
        core_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
        log_softmax = nn.LogSoftmax(dim=1)

        assert kl_target.dtype != torch.int64
        loss = core_criterion(log_softmax(input), kl_target)
        return loss

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'],
                                nesterov=config['nesterov'])

    return criterion, optimizer



def get_data_creator(config): # loads cifar10
    # for loading imagenet check make_dataloader_prev
    """ Returns train/val distributed dataloaders (cf. ImageNet in 1hr) """

    data_dir = config['dataset_dir']
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    # log.debug('fpaths train {}'.format(train_dir))
    transform1 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        normalize])


    train_dataset = datasets.ImageFolder(
            train_dir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                normalize]))

    # train_dataset = datasets.CIFAR10(train_dir, train=True, transform=transform1, target_transform=None, download=True)
    # train_dataset = datasets.CIFAR10(val_dir, train=False, transform=transform1, target_transform=None, download=True) # TODO

    # # sampler produces indices used to assign each agent data samples
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #                     dataset=train_dataset,
    #                     num_replicas=args.world_size,
    #                     rank=args.rank)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size,
    #     shuffle=(train_sampler is None),
    #     num_workers=args.num_dataloader_workers,
    #     pin_memory=True, sampler=train_sampler)

    # ret = ( train_loader, train_sampler)


    # log.debug('fpaths val {}'.format(val_dir))

    transform1=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    # val_dataset = datasets.CIFAR10(val_dir, train=False, transform=transform1, target_transform=None, download=True)

    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize]))

        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset,
        #     batch_size=args.batch_size, shuffle=False,
        #     num_workers=args.num_dataloader_workers, pin_memory=True)

        # ret = (val_loader)

    # print('Generated DataLoader')

    return train_dataset, val_dataset



def get_model_creator(config):
    """
    Initialize resnet50 similarly to "ImageNet in 1hr" paper
        Batch norm moving average "momentum" <-- 0.9
        Fully connected layer <-- Gaussian weights (mean=0, std=0.01)
        gamma of last Batch norm layer of each residual block <-- 0
    """
    model = models.resnet50()
    # model = models.resnet18()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            num_features = m.bn3.num_features
            m.bn3.weight = Parameter(torch.zeros(num_features))
    model.fc.weight.data.normal_(0, 0.01)
    model.cuda()
    # if args.fp16 and not args.amp:
    #     model = network_to_half(model)
    return model




def train(config, model, criterion, optimizer, batch_meter, data_meter, nn_meter,
          loader, epoch, itr, begin_time, num_itr_ignore, log):

    losses = Meter(ptag='Loss')
    top1 = Meter(ptag='Prec@1')
    top5 = Meter(ptag='Prec@5')

    # switch to train mode
    model.train()

    # spoof sampler to continue from checkpoint w/o loading data all over again
    _train_loader = loader.__iter__()
    for i in range(itr):
        try:
            next(_train_loader.sample_iter)
        except Exception:
            # finished epoch but prempted before state was updated
            log.info('Loader spoof error attempt {}/{}'.format(i, len(loader)))
            return

    log.debug('Training (epoch {})'.format(epoch))

    batch_time = time.time()
    for i, (batch, target) in enumerate(_train_loader, start=itr):
        # if args.fp16:
        #     batch = batch.cuda(non_blocking=True).half()

        target = target.cuda(non_blocking=True)
        # create one-hot vector from target
        kl_target = torch.zeros(target.shape[0], 1000, device='cuda').scatter_(
            1, target.view(-1, 1), 1)

        if num_itr_ignore == 0:
            data_meter.update(time.time() - batch_time)

        # ----------------------------------------------------------- #
        # Forward/Backward pass
        # ----------------------------------------------------------- #
        nn_time = time.time()
        output = model(batch)
        loss = criterion(output, kl_target)

        # if args.fp16:
        #     if args.amp:
        #         with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
        #             scaled_loss.backward()
        #     else:
        #         optimizer.backward(loss)
        # else:
        #     loss.backward()

        loss.backward()

        if i % 100 == 0:
            update_learning_rate(config, optimizer, epoch, log, itr=i,
                                 itr_per_epoch=len(loader))
        optimizer.step()  # optimization update
        optimizer.zero_grad()
        if not config['overlap'] and not config['all_reduce']:
            log.debug('Transferring params')
            model.transfer_params()
        if num_itr_ignore == 0:
            nn_meter.update(time.time() - nn_time)
        # ----------------------------------------------------------- #

        if num_itr_ignore == 0:
            batch_meter.update(time.time() - batch_time)
        batch_time = time.time()

        log_time = time.time()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), batch.size(0))
        top1.update(prec1.item(), batch.size(0))
        top5.update(prec5.item(), batch.size(0))
        if i % config['print_freq'] == 0:
            with open(config['out_fname'], '+a') as f:
                print('{ep},{itr},{bt},{nt},{dt},'
                      '{loss.val:.4f},{loss.avg:.4f},'
                      '{top1.val:.3f},{top1.avg:.3f},'
                      '{top5.val:.3f},{top5.avg:.3f},-1'
                      .format(ep=epoch, itr=i,
                              bt=batch_meter,
                              dt=data_meter, nt=nn_meter,
                              loss=losses, top1=top1,
                              top5=top5), file=f)
        if num_itr_ignore > 0:
            num_itr_ignore -= 1
        log_time = time.time() - log_time
        log.debug(log_time)

        if (config['num_iterations_per_training_epoch'] != -1 and
                i+1 == config['num_iterations_per_training_epoch']):
            break

    with open(config['out_fname'], '+a') as f:
        print('{ep},{itr},{bt},{nt},{dt},'
              '{loss.val:.4f},{loss.avg:.4f},'
              '{top1.val:.3f},{top1.avg:.3f},'
              '{top5.val:.3f},{top5.avg:.3f},-1'
              .format(ep=epoch, itr=i,
                      bt=batch_meter,
                      dt=data_meter, nt=nn_meter,
                      loss=losses, top1=top1,
                      top5=top5), file=f)

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, log):
    """ Evaluate model using criterion on validation set """

    losses = Meter(ptag='Loss')
    top1 = Meter(ptag='Prec@1')
    top5 = Meter(ptag='Prec@5')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (features, target) in enumerate(val_loader):

            # if args.fp16:
            #     features = features.cuda(non_blocking=True).half()
                # This is not needed but let it be since there is no harm

            target = target.cuda(non_blocking=True)
            # create one-hot vector from target
            kl_target = torch.zeros(
                target.shape[0], 1000, device='cuda').scatter_(
                    1, target.view(-1, 1), 1)

            # compute output
            output = model(features)
            loss = criterion(output, kl_target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), features.size(0))
            top1.update(prec1.item(), features.size(0))
            top5.update(prec5.item(), features.size(0))

        # log.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        #          .format(top1=top1, top5=top5))
        log.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.3f}'
                 .format(top1=top1, top5=top5, losses = losses))

        # print('pp * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.3f}'
        #          .format(top1=top1, top5=top5, losses = losses))

    return losses.avg, top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_state(state, update_dict):
    """ Helper function to update global state dict """
    for key in update_dict:
        state[key] = copy.deepcopy(update_dict[key])


def update_peers_per_itr(config,model, epoch):
    """ Update the model's peers per itr according to specified schedule """
    ppi = None
    e_max = -1
    for e in config['ppi_schedule']:
        if e_max <= e and epoch >= e:
            e_max = e
            ppi = config['ppi_schedule'][e]
    model.update_gossiper('peers_per_itr', ppi)


def update_learning_rate(config, optimizer, epoch, log, itr=None, itr_per_epoch=None,
                         scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    target_lr = config['lr'] * config['batch_size'] * scale * config['world_size'] / 256

    lr = None
    if config['warmup'] and epoch < 5:  # warmup to scaled lr
        if target_lr <= config['lr']:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - config['lr']) * (count / (5 * itr_per_epoch))
            lr = config['lr'] + incr
    else:
        lr = target_lr
        for e in config['lr_schedule']:
            if epoch >= e:
                lr *= config['lr_schedule'][e]

    if lr is not None:
        log.debug('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def make_logger2(verbose=True):
    # same as experiemnt_utils.make_logger except it does not have rank info
    """
    Return a logger for writing to stdout; only one logger for each application
    Arguments:
        rank (int): rank of node making logger
        verbose (bool): whether to set log-level to INFO; o.w. WARNING
    Returns:
        Python logger
    """
    logger = logging.getLogger(__name__)
    if not getattr(logger, 'handler_set', None):
        console = logging.StreamHandler(stream=sys.stdout)
        format_str = '%(levelname)s -- %(threadName)s -- %(message)s'
        console.setFormatter(logging.Formatter(format_str))
        logger.addHandler(console)  # prints to console
        logger.handler_set = True
    if not getattr(logger, 'level_set', None):
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        logger.level_set = True
    return logger


