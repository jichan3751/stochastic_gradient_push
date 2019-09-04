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

from apex import amp
from apex.parallel import DistributedDataParallel as ApexDDP
from apex.fp16_utils import network_to_half, FP16_Optimizer
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

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Playground')
parser.add_argument('--all_reduce', default='False', type=str,
                    help='whether to use all-reduce or gossip')
parser.add_argument('--batch_size', default=32, type=int,
                    help='per-agent batch size')
parser.add_argument('--lr', default=0.1, type=float,
                    help='reference learning rate (for 256 sample batch-size)')
parser.add_argument('--num_dataloader_workers', default=10, type=int,
                    help='number of dataloader workers to fork from main')
parser.add_argument('--num_epochs', default=90, type=int,
                    help='number of epochs to train')
parser.add_argument('--num_iterations_per_training_epoch', default=None,
                    type=int, help='number of iterations to run in the '
                    'training loop. To be used only for testing - to allow '
                    'training loop to exit early. None indicates that the '
                    'number of iterations per epoch will be '
                    '(num training instances)/(batch size)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='optimization momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='regularization applied to non batch-norm weights')
parser.add_argument('--nesterov', default='False', type=str,
                    help='whether to use nesterov style momentum'
                         'otherwise will use regular Polyak momentum')
parser.add_argument('--push_sum', default='True', type=str,
                    help='whether to use push-sum or push-pull gossip')
parser.add_argument('--graph_type', default=5, type=int,
                    choices=GRAPH_TOPOLOGIES,
                    help='the graph topology to use for gossip'
                         'cf. the gossip_module graph_manager for available'
                         'graph topologies and their corresponding int-id')
parser.add_argument('--mixing_strategy', default=0, type=int,
                    choices=MIXING_STRATEGIES,
                    help='the mixing strategy to use for gossip'
                         'cf. the gossip_module mixing_manager for available'
                         'mixing strategies and their corresponding int-id.')
parser.add_argument('--schedule', nargs='+', default='30 0.1 60 0.1 80 0.1',
                    type=float, help='learning rate schedule')
parser.add_argument('--peers_per_itr_schedule', nargs='+', type=int,
                    help='epoch schedule of num peers to send msgs to;'
                         'the expected format is list[epoch, num_peers]'
                         'if manually specifying peers_per_itr_schedule,'
                         'you must specify num_peers at epoch 0; i.e.,'
                         'list must contain: 0, __num_peers_at_epoch_0__')
parser.add_argument('--overlap', default='False', type=str,
                    help='whether to overlap communication with computation')
parser.add_argument('--synch_freq', default=0, type=int,
                    help='max number of iterations to go without synchronizing'
                         'communication between nodes')
parser.add_argument('--warmup', default='False', type=str,
                    help='whether to warmup learning rate for first 5 epochs')
parser.add_argument('--seed', default=47, type=int,
                    help='seed used for ALL stochastic elements in script')
parser.add_argument('--resume', default='False', type=str,
                    help='whether to resume from previously saved checkpoint')
parser.add_argument('--backend', default='nccl',
                    choices=['nccl', 'gloo', 'mpi'],
                    help='torch.distributed backend')
parser.add_argument('--tag', default='', type=str,
                    help='tag used to prepend checkpoint file names')
parser.add_argument('--print_freq', default=10, type=int,
                    help='frequency (itr.) with which to print train stats')
parser.add_argument('--verbose', default='True', type=str,
                    help='whether to log everything or just warnings/errors')
parser.add_argument('--train_fast', default='False', type=str,
                    help='whether to run script with only one validation run'
                         '(at the end once the model is trained)')
parser.add_argument('--checkpoint_all', default='True', type=str,
                    help='True: save each agents model at each epoch'
                         'False: save just one (rank 0) model at each epoch')
parser.add_argument('--overwrite_checkpoints', default='True', type=str,
                    help='True: save checkpoint at each epoch with unique tag'
                         'False: overwrite checkpoints from one epoch to next')
parser.add_argument('--master_port', default='40100', type=str,
                    help='port used to initialize distributed backend')
parser.add_argument('--checkpoint_dir', type=str,
                    help='directory for saving log-files')
parser.add_argument('--network_interface_type', default='infiniband',
                    choices=['infiniband', 'ethernet'],
                    help='network interface type to be used for communication')
parser.add_argument('--fp16', action='store_true',
                    help='whether to enable floating point 16 for speedup')
parser.add_argument('--amp', action='store_true',
                    help='whether to use apex amp for converting to fp16 '
                    'instead of FP16Optimizer')
parser.add_argument('--apex_ddp', action='store_true',
                    help='whether to use DistributedDataParallel by Apex')
parser.add_argument('--num_itr_ignore', type=int, default=10,
                    help='number of iterations to ignore before timing. A '
                         'value of 0 would imply no iterations should be '
                         'ignored')
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--no_cuda_streams', action='store_true',
                    help='do not use multiple cuda streams which are used to '
                    'speed up gossiping')
# --------------------------------------------------------------------------- #


def main():

    global amp_handle, args, state, log
    args = parse_args()
    if args.fp16 and args.amp:
        amp_handle = amp.init()
    print('parsed_args')
    print(args)

    log = make_logger(args.rank, args.verbose)
    log.info('args: {}'.format(args))
    log.info(socket.gethostname())

    # seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # init model, loss, and optimizer
    model = init_model()
    if args.all_reduce:
        if args.fp16 and args.apex_ddp:
            model = ApexDDP(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = GossipDataParallel(model,
                                   graph=args.graph,
                                   mixing=args.mixing,
                                   comm_device=args.comm_device,
                                   push_sum=args.push_sum,
                                   overlap=args.overlap,
                                   synch_freq=args.synch_freq,
                                   verbose=args.verbose,
                                   use_streams=not args.no_cuda_streams)

    core_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
    log_softmax = nn.LogSoftmax(dim=1)

    def criterion(input, kl_target):
        assert kl_target.dtype != torch.int64
        loss = core_criterion(log_softmax(input), kl_target)
        return loss

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    if args.fp16 and not args.amp:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True,
                                   verbose=False)
    optimizer.zero_grad()

    # dictionary used to encode training state
    state = {}
    update_state(state, {
            'epoch': 0, 'itr': 0, 'best_prec1': 0, 'is_best': True,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'elapsed_time': 0,
            'batch_meter': Meter(ptag='Time').__dict__,
            'data_meter': Meter(ptag='Data').__dict__,
            'nn_meter': Meter(ptag='Forward/Backward').__dict__
    })

    # module used to relaunch jobs and handle external termination signals
    cmanager = ClusterManager(rank=args.rank,
                              world_size=args.world_size,
                              model_tag=args.tag,
                              state=state,
                              all_workers=args.checkpoint_all)

    # resume from checkpoint
    if args.resume:
        if os.path.isfile(cmanager.checkpoint_fpath):
            log.info("=> loading checkpoint '{}'"
                     .format(cmanager.checkpoint_fpath))
            checkpoint = torch.load(cmanager.checkpoint_fpath)
            update_state(state, {
                          'epoch': checkpoint['epoch'],
                          'itr': checkpoint['itr'],
                          'best_prec1': checkpoint['best_prec1'],
                          'is_best': False,
                          'state_dict': checkpoint['state_dict'],
                          'optimizer': checkpoint['optimizer'],
                          'elapsed_time': checkpoint['elapsed_time'],
                          'batch_meter': checkpoint['batch_meter'],
                          'data_meter': checkpoint['data_meter'],
                          'nn_meter': checkpoint['nn_meter']
            })
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {}; itr {})"
                     .format(cmanager.checkpoint_fpath,
                             checkpoint['epoch'], checkpoint['itr']))
        else:
            log.info("=> no checkpoint found at '{}'"
                     .format(cmanager.checkpoint_fpath))

    # enable low-level optimization of compute graph using cuDNN library?
    cudnn.benchmark = True

    # meters used to compute timing stats
    batch_meter = Meter(state['batch_meter'])
    data_meter = Meter(state['data_meter'])
    nn_meter = Meter(state['nn_meter'])

    # initalize log file
    if not os.path.exists(args.out_fname):
        with open(args.out_fname, 'w') as f:
            print('BEGIN-TRAINING\n'
                  'World-Size,{ws}\n'
                  'Num-DLWorkers,{nw}\n'
                  'Batch-Size,{bs}\n'
                  'Epoch,itr,BT(s),avg:BT(s),std:BT(s),'
                  'NT(s),avg:NT(s),std:NT(s),'
                  'DT(s),avg:DT(s),std:DT(s),'
                  'Loss,avg:Loss,Prec@1,avg:Prec@1,Prec@5,avg:Prec@5,val'
                  .format(ws=args.world_size,
                          nw=args.num_dataloader_workers,
                          bs=args.batch_size), file=f)

    # create distributed data loaders
    loader, sampler = make_dataloader(args, train=True)
    if not args.train_fast:
        val_loader = make_dataloader(args, train=False)

    start_itr = state['itr']
    start_epoch = state['epoch']
    elapsed_time = state['elapsed_time']
    begin_time = time.time() - state['elapsed_time']
    best_val_prec1 = 0
    for epoch in range(start_epoch, args.num_epochs):

        # deterministic seed used to load agent's subset of data
        sampler.set_epoch(epoch + args.seed * 90)

        if not args.all_reduce:
            # update the model's peers_per_itr attribute
            update_peers_per_itr(model, epoch)

        # start all agents' training loop at same time
        if not args.all_reduce:
            model.block()
        train(model, criterion, optimizer,
              batch_meter, data_meter, nn_meter,
              loader, epoch, start_itr, begin_time, args.num_itr_ignore)

        start_itr = 0
        if not args.train_fast:
            # update state after each epoch
            elapsed_time = time.time() - begin_time
            update_state(state, {
                'epoch': epoch + 1, 'itr': start_itr,
                'is_best': False,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'elapsed_time': elapsed_time,
                'batch_meter': batch_meter.__dict__,
                'data_meter': data_meter.__dict__,
                'nn_meter': nn_meter.__dict__
            })
            # evaluate on validation set and save checkpoint
            prec1 = validate(val_loader, model, criterion)
            with open(args.out_fname, '+a') as f:
                print('{ep},{itr},{bt},{nt},{dt},'
                      '{filler},{filler},'
                      '{filler},{filler},'
                      '{filler},{filler},'
                      '{val}'
                      .format(ep=epoch, itr=-1,
                              bt=batch_meter,
                              dt=data_meter, nt=nn_meter,
                              filler=-1, val=prec1), file=f)

            if prec1 > best_val_prec1:
                update_state(state, {'is_best': True})
                best_val_prec1 = prec1

            epoch_id = epoch if not args.overwrite_checkpoints else None

            cmanager.save_checkpoint(
                epoch_id, requeue_on_signal=(epoch != args.num_epochs-1))

    if args.train_fast:
        val_loader = make_dataloader(args, train=False)
        prec1 = validate(val_loader, model, criterion)
        log.info('Test accuracy: {}'.format(prec1))

    log.info('elapsed_time {0}'.format(elapsed_time))


def train(model, criterion, optimizer, batch_meter, data_meter, nn_meter,
          loader, epoch, itr, begin_time, num_itr_ignore):

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
        if args.fp16:
            batch = batch.cuda(non_blocking=True).half()

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

        if args.fp16:
            if args.amp:
                with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                optimizer.backward(loss)
        else:
            loss.backward()

        if i % 100 == 0:
            update_learning_rate(optimizer, epoch, itr=i,
                                 itr_per_epoch=len(loader))
        optimizer.step()  # optimization update
        optimizer.zero_grad()
        if not args.overlap and not args.all_reduce:
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
        if i % args.print_freq == 0:
            with open(args.out_fname, '+a') as f:
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

        if (args.num_iterations_per_training_epoch != -1 and
                i+1 == args.num_iterations_per_training_epoch):
            break

    with open(args.out_fname, '+a') as f:
        print('{ep},{itr},{bt},{nt},{dt},'
              '{loss.val:.4f},{loss.avg:.4f},'
              '{top1.val:.3f},{top1.avg:.3f},'
              '{top5.val:.3f},{top5.avg:.3f},-1'
              .format(ep=epoch, itr=i,
                      bt=batch_meter,
                      dt=data_meter, nt=nn_meter,
                      loss=losses, top1=top1,
                      top5=top5), file=f)


def validate(val_loader, model, criterion):
    """ Evaluate model using criterion on validation set """

    losses = Meter(ptag='Loss')
    top1 = Meter(ptag='Prec@1')
    top5 = Meter(ptag='Prec@5')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (features, target) in enumerate(val_loader):

            if args.fp16:
                features = features.cuda(non_blocking=True).half()
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

        log.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5))

    return top1.avg


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


def update_peers_per_itr(model, epoch):
    """ Update the model's peers per itr according to specified schedule """
    ppi = None
    e_max = -1
    for e in args.ppi_schedule:
        if e_max <= e and epoch >= e:
            e_max = e
            ppi = args.ppi_schedule[e]
    model.update_gossiper('peers_per_itr', ppi)


def update_learning_rate(optimizer, epoch, itr=None, itr_per_epoch=None,
                         scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    target_lr = args.lr * args.batch_size * scale * args.world_size / 256

    lr = None
    if args.warmup and epoch < 5:  # warmup to scaled lr
        if target_lr <= args.lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - args.lr) * (count / (5 * itr_per_epoch))
            lr = args.lr + incr
    else:
        lr = target_lr
        for e in args.lr_schedule:
            if epoch >= e:
                lr *= args.lr_schedule[e]

    if lr is not None:
        log.debug('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def make_dataloader_prev(args, train=True): #loads imagenet
    """ Returns train/val distributed dataloaders (cf. ImageNet in 1hr) """

    data_dir = args.dataset_dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if train:
        # datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)



        log.debug('fpaths train {}'.format(train_dir))
        train_dataset = datasets.ImageFolder(
            train_dir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                normalize]))

        # sampler produces indices used to assign each agent data samples
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            dataset=train_dataset,
                            num_replicas=args.world_size,
                            rank=args.rank)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_dataloader_workers,
            pin_memory=True, sampler=train_sampler)

        return train_loader, train_sampler

    else:
        log.debug('fpaths val {}'.format(val_dir))

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_dataloader_workers, pin_memory=True)

        return val_loader



def make_dataloader(args, train=True): # loads cifar10
    """ Returns train/val distributed dataloaders (cf. ImageNet in 1hr) """

    data_dir = args.dataset_dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if train:
        log.debug('fpaths train {}'.format(train_dir))
        transform1 = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                normalize])

        train_dataset = datasets.CIFAR10(train_dir, train=True, transform=transform1, target_transform=None, download=True)

        # sampler produces indices used to assign each agent data samples
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            dataset=train_dataset,
                            num_replicas=args.world_size,
                            rank=args.rank)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_dataloader_workers,
            pin_memory=True, sampler=train_sampler)

        return train_loader, train_sampler

    else:
        log.debug('fpaths val {}'.format(val_dir))

        transform1=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])

        val_dataset = datasets.CIFAR10(train_dir, train=False, transform=transform1, target_transform=None, download=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_dataloader_workers, pin_memory=True)

        return val_loader


def parse_args():
    """
    Set env-vars and global args
        rank: <-- $SLRUM_PROCID
        world_size<-- $SLURM_NTASKS
        Master address <-- $SLRUM_NODENAME of rank 0 process (or HOSTNAME)
        Master port <-- any free port (doesn't really matter)
    """
    args1 = parser.parse_args()
    ClusterManager.set_checkpoint_dir(args1.checkpoint_dir)

    # rank and world_size need to be changed depending on the scheduler being
    # used to run the distributed jobs
    args1.master_addr = os.environ['HOSTNAME']
    if args1.backend == 'mpi':
        args1.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args1.world_size = int(os.environ['OMPI_UNIVERSE_SIZE'])
    else:
        args1.rank = int(os.environ['SLURM_PROCID'])
        args1.world_size = int(os.environ['SLURM_NTASKS'])

    # import ipdb;ipdb.set_trace()

    args1.out_fname = ClusterManager.CHECKPOINT_DIR \
        + args1.tag \
        + 'out_r' + str(args1.rank) \
        + '_n' + str(args1.world_size) \
        + '.csv'
    args1.resume = True if args1.resume == 'True' else False
    args1.verbose = True if args1.verbose == 'True' else False
    args1.train_fast = True if args1.train_fast == 'True' else False
    args1.nesterov = True if args1.nesterov == 'True' else False
    args1.checkpoint_all = True if args1.checkpoint_all == 'True' else False
    args1.warmup = True if args1.warmup == 'True' else False
    args1.overlap = True if args1.overlap == 'True' else False
    args1.push_sum = True if args1.push_sum == 'True' else False
    args1.all_reduce = True if args1.all_reduce == 'True' else False
    args1.cpu_comm = True if (args1.backend == 'gloo' and not args1.push_sum and
                             not args1.all_reduce) else False
    args1.comm_device = torch.device('cpu') if args1.cpu_comm else torch.device('cuda')
    args1.overwrite_checkpoints = True if args1.overwrite_checkpoints == 'True' else False
    args1.lr_schedule = {}
    if args1.schedule is None:
        args1.schedule = [30, 0.1, 60, 0.1, 80, 0.1]
    i, epoch = 0, None
    for v in args1.schedule:
        if i == 0:
            epoch = v
        elif i == 1:
            args1.lr_schedule[epoch] = v
        i = (i + 1) % 2
    del args1.schedule

    # parse peers per itr sched (epoch, num_peers)
    args1.ppi_schedule = {}
    if args1.peers_per_itr_schedule is None:
        args1.peers_per_itr_schedule = [0, 1]
    i, epoch = 0, None
    for v in args1.peers_per_itr_schedule:
        if i == 0:
            epoch = v
        elif i == 1:
            args1.ppi_schedule[epoch] = v
        i = (i + 1) % 2
    del args1.peers_per_itr_schedule
    # must specify how many peers to communicate from the start of training
    assert 0 in args1.ppi_schedule

    if args1.all_reduce:
        assert args1.graph_type == -1

    if args1.fp16:
        assert args1.backend == 'nccl'

    if args1.backend == 'gloo':
        assert args1.network_interface_type == 'ethernet'
        os.environ['GLOO_SOCKET_IFNAME'] = get_tcp_interface_name(
            network_interface_type=args1.network_interface_type
        )
    elif args1.network_interface_type == 'ethernet':
        if args1.backend == 'nccl':
            os.environ['NCCL_SOCKET_IFNAME'] = get_tcp_interface_name(
                network_interface_type=args1.network_interface_type
            )
            os.environ['NCCL_IB_DISABLE'] = '1'
        else:
            raise NotImplementedError

    # initialize torch distributed backend
    os.environ['MASTER_ADDR'] = args1.master_addr
    os.environ['MASTER_PORT'] = args1.master_port
    dist.init_process_group(backend=args1.backend,
                            world_size=args1.world_size,
                            rank=args1.rank)

    args1.graph, args1.mixing = None, None
    graph_class = GRAPH_TOPOLOGIES[args1.graph_type]
    if graph_class:
        # dist.barrier is done here to ensure the NCCL communicator is created
        # here. This prevents an error which may be caused if the NCCL
        # communicator is created at a time gap of more than 5 minutes in
        # different processes
        dist.barrier()
        args1.graph = graph_class(
            args1.rank, args1.world_size, peers_per_itr=args1.ppi_schedule[0])

    mixing_class = MIXING_STRATEGIES[args1.mixing_strategy]
    if mixing_class and args1.graph:
        args1.mixing = mixing_class(args1.graph, args1.comm_device)

    return args1


def init_model():
    """
    Initialize resnet50 similarly to "ImageNet in 1hr" paper
        Batch norm moving average "momentum" <-- 0.9
        Fully connected layer <-- Gaussian weights (mean=0, std=0.01)
        gamma of last Batch norm layer of each residual block <-- 0
    """
    # model = models.resnet50()
    model = models.resnet18()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            num_features = m.bn3.num_features
            m.bn3.weight = Parameter(torch.zeros(num_features))
    model.fc.weight.data.normal_(0, 0.01)
    model.cuda()
    if args.fp16 and not args.amp:
        model = network_to_half(model)
    return model


if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True)
    main()
