from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
import socket
import time
import logging

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
import ray
from ray.experimental.sgd.pytorch import pytorch_utils
from ray.experimental.sgd import utils

import ray_sgp_utils as sgp_utils
from ray_sgp_utils import update_state, GRAPH_TOPOLOGIES, MIXING_STRATEGIES

logger = sgp_utils.make_logger2(verbose=False) # same as sgp code
# logger.setLevel(logging.DEBUG)

def main(config):

    logger.info('config: {}'.format(config))
    logger.info(socket.gethostname())

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True

    sgp_runner = SGPRunner(
        model_creator=sgp_utils.get_model_creator,
        data_creator=sgp_utils.get_data_creator,
        optimizer_creator = sgp_utils.get_optimizer_creator,
        config = config
        )


    address = "tcp://{ip}:{port}".format(ip=config['master_addr'], port=config['master_port'])
    sgp_runner.setup(address, config['rank'], config['world_size'])

    stat1 = sgp_runner.step()
    print(stat1)

    stat1 = sgp_runner.step()
    print(stat1)

    # train_stats = sgp_runner.step()
    # print(train_stats)
    # train_stats = sgp_runner.step()
    # print(train_stats)

    # val_stats = sgp_runner.validate()
    # print(val_stats)




class SGPRunner(object):

    def __init__(self,
                 model_creator,
                 data_creator,
                 optimizer_creator,
                 config=None):
        """Initializes the runner.

        Args:
            model_creator (dict -> torch.nn.Module): see pytorch_trainer.py.
            data_creator (dict -> Dataset, Dataset): see pytorch_trainer.py.
            optimizer_creator (torch.nn.Module, dict -> loss, optimizer):
                see pytorch_trainer.py.
            config (dict): see pytorch_trainer.py.
            batch_size (int): see pytorch_trainer.py.
        """

        self.model_creator = model_creator
        self.data_creator = data_creator
        self.optimizer_creator = optimizer_creator
        self.config = {} if config is None else config
        self.verbose = True

        self.epoch = 0

        self._timers = {
            k: utils.TimerStat(window_size=1)
            for k in [
                "setup_proc", "setup_model", "get_state", "set_state",
                "validation", "training"
            ]
        }

    def setup(self, url, world_rank, world_size):
        """Connects to the distributed PyTorch backend and initializes the model.

        Args:
            url (str): the URL used to connect to distributed PyTorch.
            world_rank (int): the index of the runner.
            world_size (int): the total number of runners.
        """
        # print('_setup_distributed_pytorch')

        # checking current dir
        # from subprocess import Popen, PIPE
        # process = Popen(['ls', './'], stdout=PIPE, stderr=PIPE)
        # stdout, stderr = process.communicate()
        # print(stdout)

        self._update_config( world_rank, world_size)

        self._setup_distributed_pytorch(url, world_rank, world_size)
        # print('_setup_gossip_related')
        self._setup_gossip_related()
        # print('_setup_training')
        self._setup_training()
        # print('_setup_misc')
        self._setup_misc()
        # print('setup done')

    def _update_config(self, world_rank, world_size):
        self.config['rank'] = world_rank
        self.config['world_size'] = world_size
        self.config['out_fname'] = '/home/ubuntu/stochastic_gradient_push/ckpt/{tag}out_r{rank}_n{wsize}.csv'.format(
        tag = self.config['tag'], rank = self.config['rank'], wsize = self.config['world_size'])


    def _setup_distributed_pytorch(self, url, world_rank, world_size):
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # the distributed pytorch runner has this but don't know why
        with self._timers["setup_proc"]:
            self.world_rank = world_rank
            logger.debug(
                "Connecting to {} world_rank: {} world_size: {}".format(
                    url, world_rank, world_size))
            logger.debug("using {}".format(self.config['backend']))


            print("Connecting to {} world_rank: {} world_size: {}".format(
                    url, world_rank, world_size))

            dist.init_process_group(
                backend=self.config['backend'],
                init_method=url,
                rank=world_rank,
                world_size=world_size)

    def _setup_gossip_related(self):
        config = self.config
        self.comm_device = torch.device('cpu') if config['cpu_comm'] else torch.device('cuda')

        self.graph, self.mixing = None, None
        graph_class = GRAPH_TOPOLOGIES[config['graph_type']]
        if graph_class:
            # dist.barrier is done here to ensure the NCCL communicator is created
            # here. This prevents an error which may be caused if the NCCL
            # communicator is created at a time gap of more than 5 minutes in
            # different processes
            dist.barrier()
            self.graph = graph_class(
                config['rank'], config['world_size'], peers_per_itr=config['ppi_schedule'][0])

        mixing_class = MIXING_STRATEGIES[config['mixing_strategy']]
        if mixing_class and self.graph:
            self.mixing = mixing_class(self.graph, self.comm_device)



    def _setup_training(self):
        config = self.config
        ## note: assume gpu available

        logger.debug("Creating model")
        # print('model')
        self.model = self.model_creator(self.config)


        if config['all_reduce']:
            # print('DistributedDataParallel')
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        else:
            # print('GossipDataParallel')
            self.model = GossipDataParallel(self.model,
                                       graph=self.graph,
                                       mixing=self.mixing,
                                       comm_device=self.comm_device,
                                       push_sum=config['push_sum'],
                                       overlap=config['overlap'],
                                       synch_freq=config['synch_freq'],
                                       verbose=config['verbose'],
                                       use_streams=not config['no_cuda_streams'])


        logger.debug("Creating optimizer")

        # print('Optimizer')
        self.criterion, self.optimizer = self.optimizer_creator(
            self.model, self.config)
        # if torch.cuda.is_available():
        #     self.criterion = self.criterion.cuda() # ptorch runner has this but does not work here

        logger.debug("Creating dataset")

        # print('DataSet')
        self.training_set, self.validation_set = self.data_creator(self.config)

        # TODO: make num_workers configurable
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset = self.training_set,
            num_replicas=config['world_size'],
            rank=config['rank'])
        self.train_loader = torch.utils.data.DataLoader(
            self.training_set,
            batch_size=config['batch_size'],
            shuffle=(self.train_sampler is None),
            num_workers=config['num_dataloader_workers'],
            pin_memory=True,
            sampler=self.train_sampler)

        self.validation_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset = self.validation_set,
                num_replicas=config['world_size'],
                rank=config['rank'])

        # pytorch runner ver
        # self.validation_loader = torch.utils.data.DataLoader(
        #     self.validation_set,
        #     batch_size=config['batch_size'],
        #     shuffle=(self.validation_sampler is None),
        #     num_workers=config['num_dataloader_workers'],
        #     pin_memory=True,
        #     sampler=self.validation_sampler)

        # sgp code ver
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_dataloader_workers'],
            pin_memory=True)


        self.optimizer.zero_grad()



    def _setup_misc(self):
        # misc setup components that were in goissip_sgd
        config = self.config
        state = {}
        update_state(state, {
                'epoch': 0, 'itr': 0, 'best_prec1': 0, 'is_best': True,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'elapsed_time': 0,
                'batch_meter': Meter(ptag='Time').__dict__,
                'data_meter': Meter(ptag='Data').__dict__,
                'nn_meter': Meter(ptag='Forward/Backward').__dict__
        })
        self.state = state



        # module used to relaunch jobs and handle external termination signals
        ClusterManager.set_checkpoint_dir(config['checkpoint_dir'])
        self.cmanager = ClusterManager(rank=config['rank'],
                              world_size=config['world_size'],
                              model_tag=config['tag'],
                              state=state,
                              all_workers=config['checkpoint_all'])


        # enable low-level optimization of compute graph using cuDNN library?
        cudnn.benchmark = True

        self.batch_meter = Meter(state['batch_meter'])
        self.data_meter = Meter(state['data_meter'])
        self.nn_meter = Meter(state['nn_meter'])


        # initalize log file
        if not os.path.exists(config['out_fname']):
            with open(config['out_fname'], 'w') as f:
                print('BEGIN-TRAINING\n'
                      'World-Size,{ws}\n'
                      'Num-DLWorkers,{nw}\n'
                      'Batch-Size,{bs}\n'
                      'Epoch,itr,BT(s),avg:BT(s),std:BT(s),'
                      'NT(s),avg:NT(s),std:NT(s),'
                      'DT(s),avg:DT(s),std:DT(s),'
                      'Loss,avg:Loss,Prec@1,avg:Prec@1,Prec@5,avg:Prec@5,val'
                      .format(ws=config['world_size'],
                              nw=config['num_dataloader_workers'],
                              bs=config['batch_size']), file=f)


        self.start_itr = state['itr']
        self.start_epoch = state['epoch']
        self.elapsed_time = state['elapsed_time']
        self.begin_time = time.time() - state['elapsed_time']
        self.best_val_prec1 = 0


    def resume(self):

        # TODO
        pass



# neeed  optimizer.zero_grad()

    def step(self):

        config = self.config
        state = self.state


        # TODO: epoch setting?

        """Runs a training epoch and updates the model parameters."""
        logger.debug("Starting Epoch {}".format(self.epoch))

        self.train_sampler.set_epoch(self.epoch + self.config['seed'] * 90)

        if not config['all_reduce']:
            # update the model's peers_per_itr attribute
            sgp_utils.update_peers_per_itr(self.config, self.model, self.epoch)


        # start all agents' training loop at same time
        if not config['all_reduce']:
            self.model.block()

        losses_avg, top1_avg, top5_avg = sgp_utils.train(self.config, self.model, self.criterion, self.optimizer,
            self.batch_meter, self.data_meter, self.nn_meter,
            self.train_loader, self.epoch, self.start_itr, self.begin_time, self.config['num_itr_ignore'], logger)


        train_stats = {
            "epoch" : self.epoch,
            "train_loss": losses_avg,
            "train_top1": top1_avg,
            "train_top5": top5_avg,
            }


        start_itr = 0

        if not config['train_fast']:
            # update state after each epoch
            elapsed_time = time.time() - self.begin_time
            update_state(state, {
                'epoch': self.epoch + 1, 'itr': self.start_itr,
                'is_best': False,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'elapsed_time': elapsed_time,
                'batch_meter': self.batch_meter.__dict__,
                'data_meter': self.data_meter.__dict__,
                'nn_meter': self.nn_meter.__dict__
            })
            # evaluate on validation set and save checkpoint
            loss1, prec1, prec5 = sgp_utils.validate(self.validation_loader, self.model, self.criterion, logger)
            with open(config['out_fname'], '+a') as f:
                print('{ep},{itr},{bt},{nt},{dt},'
                      '{filler},{filler},'
                      '{filler},{filler},'
                      '{filler},{filler},'
                      '{val}'
                      .format(ep=self.epoch, itr=-1,
                              bt=self.batch_meter,
                              dt=self.data_meter, nt=self.nn_meter,
                              filler=-1, val=prec1), file=f)

            if prec1 > self.best_val_prec1:
                update_state(state, {'is_best': True})
                self.best_val_prec1 = prec1

            epoch_id = self.epoch if not config['overwrite_checkpoints'] else None

            self.cmanager.save_checkpoint(
                epoch_id, requeue_on_signal=(self.epoch != config['num_epochs']-1))
            print('Finished Epoch {ep}, elapsed {tt:.3f}sec'.format(ep=self.epoch,tt=elapsed_time ))


        else:
            elapsed_time = time.time() - self.begin_time
            print('Finished Epoch {ep}, elapsed {tt:.3f}sec'.format(ep=self.epoch,tt=elapsed_time ))


        self.epoch += 1

        return train_stats




#     def validate(self):
#         """Evaluates the model on the validation data set."""
#         with self._timers["validation"]:
#             validation_stats = pytorch_utils.validate(
#                 self.validation_loader, self.model, self.criterion)

#         validation_stats.update(self.stats())
#         return validation_stats

#     def stats(self):
#         """Returns a dictionary of statistics collected."""
#         stats = {"epoch": self.epoch}
#         for k, t in self._timers.items():
#             stats[k + "_time_mean"] = t.mean
#             stats[k + "_time_total"] = t.sum
#             t.reset()
#         return stats

#     def get_state(self):
#         """Returns the state of the runner."""
#         return {
#             "epoch": self.epoch,
#             "model": self.model.state_dict(),
#             "optimizer": self.optimizer.state_dict(),
#             "stats": self.stats()
#         }

#     def set_state(self, state):
#         """Sets the state of the model."""
#         # TODO: restore timer stats
#         self.model.load_state_dict(state["model"])
#         self.optimizer.load_state_dict(state["optimizer"])
#         self.epoch = state["stats"]["epoch"]

    def shutdown(self):
        """Attempts to shut down the worker."""
        del self.validation_loader
        del self.validation_set
        del self.train_loader
        del self.training_set
        del self.criterion
        del self.optimizer
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dist.destroy_process_group()



    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return utils.find_free_port()

def get_config(RANK, WSIZE, MASTER_ADDR, TASK ):

    ## TODO: parse_args

    if TASK ==0 :
        config = {'all_reduce': True, 'batch_size': 256, 'lr': 0.1, 'num_dataloader_workers': 4,
            'num_epochs': 2, 'num_iterations_per_training_epoch': None, 'momentum': 0.9, 'weight_decay': 0.0001,
            'nesterov': True, 'push_sum': False, 'graph_type': -1, 'mixing_strategy': 0, 'overlap': False,
            'synch_freq': 0, 'warmup': True, 'seed': 1, 'resume': False, 'backend': 'nccl', 'tag': 'AR-SGD-ETH',
            'print_freq': 100, 'verbose': False, 'train_fast': False, 'checkpoint_all': True, 'overwrite_checkpoints': True,
            'master_port': '40100', 'checkpoint_dir': '/home/ubuntu/sgp_ray/stochastic_gradient_push/ckpt', 'network_interface_type': 'ethernet',
            'fp16': False, 'num_itr_ignore': 10, 'dataset_dir': '/home/ubuntu/FF/datasets/cifar10',
            'no_cuda_streams': False, 'master_addr': '172.31.91.171', 'rank': 0, 'world_size': 2,
            'out_fname': './ckpt/AR-SGD-ETHout_r0_n2.csv', 'cpu_comm': False,
            'lr_schedule': {30.0: 0.1, 60.0: 0.1, 80.0: 0.1},
            'ppi_schedule': {0: 1}
            # 'comm_device': device(type='cuda'),  'graph': None, 'mixing': None
            }

    elif TASK ==1:
        config = {'all_reduce': False, 'batch_size': 256, 'lr': 0.1, 'num_dataloader_workers': 4,
            'num_epochs': 2, 'num_iterations_per_training_epoch': None, 'momentum': 0.9, 'weight_decay': 0.0001,
            'nesterov': True, 'push_sum': False, 'graph_type': 1, 'mixing_strategy': 0, 'overlap': False, 'synch_freq': 0,
            'warmup': True, 'seed': 1, 'resume': False, 'backend': 'nccl', 'tag': 'DPSGD_ETH',
            'print_freq': 100, 'verbose': False, 'train_fast': False, 'checkpoint_all': True, 'overwrite_checkpoints': True,
            'master_port': '40100', 'checkpoint_dir': '/home/ubuntu/sgp_ray/stochastic_gradient_push/ckpt', 'network_interface_type': 'ethernet',
            'fp16': False, 'num_itr_ignore': 10, 'dataset_dir': '/home/ubuntu/FF/datasets/cifar10',
            'no_cuda_streams': False, 'master_addr': '172.31.91.171', 'rank': 0, 'world_size': 2,
            'out_fname': './ckpt/DPSGD_ETHout_r0_n2.csv', 'cpu_comm': False,
            'lr_schedule': {30.0: 0.1, 60.0: 0.1, 80.0: 0.1}, 'ppi_schedule': {0: 1}
            # 'comm_device': device(type='cuda'),
            # 'graph': <gossip_module.graph_manager.DynamicBipartiteExponentialGraph object at 0x7f5f9269e8d0>,
            # 'mixing': <gossip_module.mixing_manager.UniformMixing object at 0x7f5f926a6a58>
            }

    elif TASK ==2:
        config = {'all_reduce': False, 'batch_size': 256, 'lr': 0.1, 'num_dataloader_workers': 4,
            'num_epochs': 2, 'num_iterations_per_training_epoch': None, 'momentum': 0.9, 'weight_decay': 0.0001,
            'nesterov': True, 'push_sum': True, 'graph_type': 0, 'mixing_strategy': 0, 'overlap': False, 'synch_freq': 0,
            'warmup': True, 'seed': 1, 'resume': False, 'backend': 'nccl', 'tag': 'SGP_ETH',
            'print_freq': 100, 'verbose': False, 'train_fast': False, 'checkpoint_all': True, 'overwrite_checkpoints': True,
            'master_port': '40100', 'checkpoint_dir': '/home/ubuntu/sgp_ray/stochastic_gradient_push/ckpt', 'network_interface_type': 'ethernet',
            'fp16': False, 'num_itr_ignore': 10, 'dataset_dir': '/home/ubuntu/FF/datasets/cifar10',
            'no_cuda_streams': False, 'master_addr': '172.31.91.171', 'rank': 0, 'world_size': 2,
            'out_fname': './ckpt/SGP_ETHout_r0_n2.csv', 'cpu_comm': False,
            'lr_schedule': {30.0: 0.1, 60.0: 0.1, 80.0: 0.1}, 'ppi_schedule': {0: 1},
            # 'comm_device': device(type='cuda'),
            # 'graph': <gossip_module.graph_manager.DynamicDirectedExponentialGraph object at 0x7f62cc919908>,
            # 'mixing': <gossip_module.mixing_manager.UniformMixing object at 0x7f62cc921a90>
            }

    else:
        assert 0

    config['master_addr'] = MASTER_ADDR
    config['rank'] = RANK
    config['world_size'] = WSIZE
    # config['out_fname'] = '/home/ubuntu/stochastic_gradient_push/ckpt/{tag}out_r{rank}_n{wsize}.csv'.format(
    #     tag = config['tag'], rank = config['rank'], wsize = config['world_size'])

    config['dataset_dir'] = '/data/datasets/imagenet12'
    config['checkpoint_dir'] = '/home/ubuntu/stochastic_gradient_push/ckpt'

    print(config['tag'])


    ## overrides
    # config['overlap'] = True


    return config

if __name__ == '__main__':
    import sys
    RANK = int(sys.argv[1])
    WSIZE = int(sys.argv[2])
    MASTER_ADDR = sys.argv[3]
    TASK = int(sys.argv[4])

    config =get_config(RANK, WSIZE, MASTER_ADDR, TASK )

    main(config)


