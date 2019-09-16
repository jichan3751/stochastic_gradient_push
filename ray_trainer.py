from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import torch
import logging
import argparse

import ray

from ray.experimental.sgd.pytorch import pytorch_utils
from ray.experimental.sgd import utils

import ray_sgp_utils as sgp_utils
from ray_sgp_utils import update_state, GRAPH_TOPOLOGIES, MIXING_STRATEGIES

from ray_runner import SGPRunner, get_config


logger = logging.getLogger(__name__)

def main(config, num_replicas=1, use_gpu=1):

    trainer1 = SGPTrainer(
        model_creator=sgp_utils.get_model_creator,
        data_creator=sgp_utils.get_data_creator,
        optimizer_creator = sgp_utils.get_optimizer_creator,
        num_replicas=num_replicas,
        config = config)
    trainer1.train()

    trainer1.train()

    trainer1.shutdown()
    print("success!")

    pass


class SGPTrainer(object):
    """Train a PyTorch model using distributed PyTorch.

    Launches a set of actors which connect via distributed PyTorch and
    coordinate gradient updates to train the provided model.
    """

    def __init__(self,
                 model_creator,
                 data_creator,
                 optimizer_creator=pytorch_utils.sgd_mse_optimizer,
                 config=None,
                 use_gpu = 1,
                 num_replicas=2):
        """Sets up the PyTorch trainer.

        Args:
            model_creator (dict -> torch.nn.Module): creates the model
                using the config.
            data_creator (dict -> Dataset, Dataset): creates the training
                and validation data sets using the config.
            optimizer_creator (torch.nn.Module, dict -> loss, optimizer):
                creates the loss and optimizer using the model and the config.
            config (dict): configuration passed to 'model_creator',
                'data_creator', and 'optimizer_creator'.
            num_replicas (int): the number of workers used in distributed
                training.
            use_gpu (bool): Sets resource allocation for workers to 1 GPU
                if true.
            batch_size (int): batch size for an update.
            backend (string): backend used by distributed PyTorch.
        """
        # TODO: add support for mixed precision
        # TODO: add support for callbacks
        assert num_replicas > 1
        # if num_replicas > 1 and not dist.is_available():
        #     raise ValueError(
        #         ("Distributed PyTorch is not supported on macOS. "
        #          "To run without distributed PyTorch, set 'num_replicas=1'. "
        #          "For more information, see "
        #          "https://github.com/pytorch/examples/issues/467."))

        self.model_creator = model_creator
        self.config = {} if config is None else config
        self.optimizer_timer = utils.TimerStat(window_size=1)

        # logger.info("Using {} as backend.".format(backend))

        if num_replicas == 1:
            # # Generate actor class
            # Runner = ray.remote(
            #     num_cpus=1, num_gpus=int(use_gpu))(PyTorchRunner)
            # # Start workers
            # self.workers = [
            #     Runner.remote(model_creator, data_creator, optimizer_creator,
            #                   self.config, batch_size)
            # ]
            # # Get setup tasks in order to throw errors on failure
            # ray.get(self.workers[0].setup.remote())
            pass
        else:
            # Geneate actor class
            Runner = ray.remote(
                num_cpus=1, num_gpus=int(use_gpu))(SGPRunner)
            # Compute batch size per replica
            # batch_size_per_replica = batch_size // num_replicas
            # if batch_size % num_replicas > 0:
            #     new_batch_size = batch_size_per_replica * num_replicas
            #     logger.warn(
            #         ("Changing batch size from {old_batch_size} to "
            #          "{new_batch_size} to evenly distribute batches across "
            #          "{num_replicas} replicas.").format(
            #              old_batch_size=batch_size,
            #              new_batch_size=new_batch_size,
            #              num_replicas=num_replicas))
            # Start workers
            self.workers = [
                Runner.remote(model_creator, data_creator, optimizer_creator,
                              self.config)
                for i in range(num_replicas)
            ]
            # Compute URL for initializing distributed PyTorch
            ip = ray.get(self.workers[0].get_node_ip.remote())
            port = ray.get(self.workers[0].find_free_port.remote())
            address = "tcp://{ip}:{port}".format(ip=ip, port=port)
            # Get setup tasks in order to throw errors on failure
            ray.get([
                worker.setup.remote(address, i, len(self.workers))
                for i, worker in enumerate(self.workers)
            ])

    def train(self):
        """Runs a training epoch."""
        with self.optimizer_timer:
            worker_stats = ray.get([w.step.remote() for w in self.workers])

        train_stats = worker_stats[0].copy()
        train_stats["train_loss"] = np.mean(
            [s["train_loss"] for s in worker_stats])
        return train_stats

    def validate(self):
        """Evaluates the model on the validation data set."""
        worker_stats = ray.get([w.validate.remote() for w in self.workers])
        validation_stats = worker_stats[0].copy()
        validation_stats["validation_loss"] = np.mean(
            [s["validation_loss"] for s in worker_stats])
        return validation_stats

    def get_model(self):
        """Returns the learned model."""
        model = self.model_creator(self.config)
        state = ray.get(self.workers[0].get_state.remote())
        model.load_state_dict(state["model"])
        return model

    def save(self, checkpoint):
        """Saves the model at the provided checkpoint.

        Args:
            checkpoint (str): Path to target checkpoint file.

        """
        state = ray.get(self.workers[0].get_state.remote())
        torch.save(state, checkpoint)
        return checkpoint

    def restore(self, checkpoint):
        """Restores the model from the provided checkpoint.

        Args:
            checkpoint (str): Path to target checkpoint file.

        """
        state = torch.load(checkpoint)
        state_id = ray.put(state)
        ray.get([worker.set_state.remote(state_id) for worker in self.workers])

    def shutdown(self):
        """Shuts down workers and releases resources."""
        for worker in self.workers:
            worker.shutdown.remote()
            worker.__ray_terminate__.remote()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redis-address",
        required=False,
        type=str,
        help="the address to use for Redis")
    parser.add_argument(
        "--num-replicas",
        "-n",
        type=int,
        default=1,
        help="Sets number of replicas for training.")

    parser.add_argument(
        "--use-gpu",
        type=int,
        default=1,
        help="Sets number of gpus for training.")

    parser.add_argument(
        "--task",
        type=int,
        default=1,
        help="sets the mode for training")

    args, _ = parser.parse_known_args()

    import ray

    ray.init(redis_address=args.redis_address)

    config = get_config(RANK=0, WSIZE=1, MASTER_ADDR='127.0.0.1', TASK = args.task )

    main(config, args.num_replicas, args.use_gpu )


