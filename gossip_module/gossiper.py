# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Gossipers

:description: Gossiper's are designed for multi-peer communication (i.e., send
              and recv from multiple peers at each ieration)
"""

import torch
import torch.distributed as dist

from .graph_manager import GraphManager
from .mixing_manager import MixingManager
from .mixing_manager import UniformMixing


class dist_backend:
    UNDEFINED = -1
    TCP = 0
    MPI = 1
    GLOO = 2
    NCCL = 3


class Gossiper(object):
    """ Generic gossip averaging object for multi-peer communication """

    def __init__(self, msg, graph, device=None, mixing=None, logger=None,
                 rank=None, world_size=None):
        """
        Initialize generic averaging class designed for multi-peer comms

        :param msg: (tensor) message used to initialize recv buffer
        :param device: (device) device on which to initialize recv buffer
        :param graph: (GraphManager) Subclass of GraphManager
        :param mixing: (MixingManager) Subclass of MixingManager
        :param logger: (python logger) module used to log results
        """

        self.logger = logger
        if rank is None or world_size is None:
            assert dist.is_initialized()
            # for now p2p communication only supported withed tcp and mpi
            assert dist._backend != dist_backend.GLOO
            assert dist._backend != dist_backend.NCCL
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        # graph topology properties
        self.rank = rank
        self.world_size = world_size
        assert isinstance(graph, GraphManager)
        self._graph_manager = graph
        self.peers_per_itr_device = torch.tensor(
            [self._graph_manager.peers_per_itr], device=device,
            dtype=msg.dtype)
        self.passive = self._graph_manager.is_passive()
        self.refresh_peers_(rotate=False)  # sets in- and out-peers attributes

        # mixing matrix
        if mixing is None:
            mixing = UniformMixing(self._graph_manager, device)
        assert isinstance(mixing, MixingManager)
        self._mixing_manager = mixing
        self.refresh_mixing_weights_()  # sets mixing-weights attribute

        # regular ==> we don't need to keep track of ps-weight explicitly
        self.regular = self._mixing_manager.is_regular()

        # msg buffers used during send/recv
        self.device = device if device is not None else msg.device
        self.out_msg_buffer = []
        self.in_msg_buffer = msg.clone().detach_().to(self.device)
        self._ps_weight = torch.ones(1).detach_().to(self.device).type(
            msg.dtype)
        # not using regular comms ==> need to communicate ps-weight
        if not self.regular:
            self.in_msg_buffer = torch.cat([self.in_msg_buffer,
                                            self.ps_weight])
        if self.device.type == 'cpu':
            try:
                self.in_msg_buffer = self.in_msg_buffer.pin_memory()
            except Exception as e:
                if self.logger is not None:
                    self.logger.error(e)
        self.placeholder = self.in_msg_buffer.clone()

        self._pending_req = None

    @property
    def ps_weight(self):
        return self._ps_weight

    @ps_weight.setter
    def ps_weight(self, v):
        self._ps_weight.data[0] = v

    @property
    def peers_per_itr(self):
        return self._graph_manager.peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v):
        self._graph_manager.peers_per_itr = v

    def refresh_peers_(self, rotate=None):
        """ Update in- and out-peers """
        if rotate is None:
            rotate = True if self._graph_manager.is_dynamic_graph() else False
        # cannot cycle peers in a static graph
        assert not (rotate and not self._graph_manager.is_dynamic_graph())
        self.out_edges, self.in_edges = self._graph_manager.get_edges(rotate)

    def refresh_mixing_weights_(self, residual_adjusted=False):
        """ Update mixing-matrix weights """
        self.mixing_weights = self._mixing_manager.get_mixing_weights(
            residual_adjusted)

    def mix_out_msg_(self, out_msg, ps_weight, residual=False):
        """ Returns a generator mixing messages on the fly """
        self.refresh_mixing_weights_(residual)
        self.ps_weight = ps_weight

        # check whether or not we need to communicate ps_weight
        if not self.regular:
            out_msg = torch.cat([out_msg, self.ps_weight.type(out_msg.dtype)])

        # first return 'loopback msg to self'
        if not residual:
            yield out_msg.mul(self.mixing_weights['lo'].type(out_msg.dtype))

        # check whether or not we need to create a buffer for each out-msg
        if self._mixing_manager.is_uniform():
            weight = self.mixing_weights['uniform']
            out_msg *= weight.type(out_msg.dtype)
            for _ in self.out_edges:
                yield out_msg
        else:
            for out_edge in self.out_edges:
                weight = self.mixing_weights[out_edge.dest]
                yield out_msg.mul(weight.type(out_msg.dtype))

    def clean_msg_buffers_(self):
        """ Clean outgoing message buffer """
        msgs = []
        while len(self.out_msg_buffer) > 0:
            req, msg = self.out_msg_buffer.pop()
            req.wait()
            msgs.append(msg)
        while len(msgs) > 0:
            msg = msgs.pop()
            # msg.data.set_() # due to change https://github.com/pytorch/pytorch/issues/23393
            with torch.no_grad():
                msg.set_()

    def parse_in_msg_buffer(self, residual=False):
        """ Parse in-msg buffer and return msg and ps-weight separately """
        msg = self.in_msg_buffer
        if not self.regular:
            return msg.narrow(0, 0, len(msg) - 1), msg[-1]
        else:
            if residual:
                return msg, self.ps_weight * self.peers_per_itr_device
            else:
                return msg, torch.ones(1, device=self.device).type(msg.dtype)

    def mix(self):
        """ Single gossip step """
        raise NotImplementedError


class PushSum(Gossiper):
    """ 1-peer Push-Sum consensus averaging module """

    def mix(self, out_msg, ps_weight, residual=False):
        """ Consensus averaging step """
        # out_msg must be on the correct device
        assert out_msg.device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'
                              .format(self.in_edges, self.out_edges))

        # prepare messages for gossip
        mixed_out_msgs = self.mix_out_msg_(out_msg, ps_weight, residual)

        # non-blocking send
        for out_edge in self.out_edges:
            msg = next(mixed_out_msgs)
            assert self.rank == out_edge.src
            req = dist.broadcast(tensor=msg, src=out_edge.src,
                                 group=out_edge.process_group, async_op=True)
            self.out_msg_buffer.append((req, msg))

        # blocking recv w/ some code optimization to avoid buffer prep overhead
        if len(self.in_edges) == 1 and residual:
            in_edge = self.in_edges[0]
            dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src,
                           group=in_edge.process_group)

        # regular non-blocking recv
        else:
            # prepare in-msg buffer
            if not residual:
                self.in_msg_buffer.copy_(next(mixed_out_msgs))
            else:
                self.in_msg_buffer.zero_()

            for in_edge in self.in_edges:
                dist.broadcast(tensor=self.placeholder, src=in_edge.src,
                               group=in_edge.process_group)
                self.in_msg_buffer.add_(self.placeholder)

        self.refresh_peers_()
        self.clean_msg_buffers_()
        return self.parse_in_msg_buffer(residual)


class PushPull(Gossiper):
    """ Doubly-stochastic consensus averaging module """

    def mix(self, out_msg, ps_weight, residual=False):
        # out_msg must be on the correct device
        assert out_msg.device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'
                              .format(self.in_edges, self.out_edges))

        # prepare messages for gossip
        mixed_out_msgs = self.mix_out_msg_(out_msg, ps_weight, residual)

        # send-recv w/ some code optimization to avoid buffer prep overhead
        if len(self.in_edges) == 1 and len(self.out_edges) == 1 and residual:
            out_edge, in_edge = self.out_edges[0], self.in_edges[0]
            msg = next(mixed_out_msgs)
            if not self.passive:
                dist.broadcast(tensor=msg, src=out_edge.src,
                               group=out_edge.process_group)
                dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src,
                               group=in_edge.process_group)
            else:
                dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src,
                               group=in_edge.process_group)
                dist.broadcast(tensor=msg, src=out_edge.src,
                               group=out_edge.process_group)

        # regular send-recv
        else:
            # prepare in-msg buffer
            if not residual:
                self.in_msg_buffer.copy_(next(mixed_out_msgs))
            else:
                self.in_msg_buffer.zero_()

            # send-recv
            for out_edge, in_edge in zip(self.out_edges, self.in_edges):
                msg = next(mixed_out_msgs)
                if not self.passive:
                    dist.broadcast(tensor=msg, src=out_edge.src,
                                   group=out_edge.process_group)
                    dist.broadcast(tensor=self.placeholder, src=in_edge.src,
                                   group=in_edge.process_group)
                else:
                    dist.broadcast(tensor=self.placeholder, src=in_edge.src,
                                   group=in_edge.process_group)
                    dist.broadcast(tensor=msg, src=out_edge.src,
                                   group=out_edge.process_group)
                self.in_msg_buffer.add_(self.placeholder)

        self.refresh_peers_()
        self.clean_msg_buffers_()
        return self.parse_in_msg_buffer(residual)


class BilatPushPull(Gossiper):
    """ Bilateral doubly-stochastic consensus averaging module """

    def mix(self, out_msg):
        # out_msg must be on the correct device
        assert out_msg.device.type == self.device.type
        # bilat gossip can only send and receive from one peer at each itr
        assert len(self.in_edges) == 1 and len(self.out_edges) == 1
        out_edge, in_edge = self.out_edges[0], self.in_edges[0]
        if self.logger is not None:
            self.logger.debug('in/out -edges {}/{}'.format(in_edge, out_edge))

        if not self.passive:
            # prepare messages for gossip
            mixed_out_msgs = self.mix_out_msg_(out_msg, 1., residual=True)
            msg = next(mixed_out_msgs)
            # blocking send/recv
            dist.broadcast(tensor=msg, src=out_edge.src,
                           group=out_edge.process_group)
            dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src,
                           group=in_edge.process_group)
            completed = True
        else:
            if self._pending_req is None:
                self._pending_req = dist.broadcast(
                    tensor=self.in_msg_buffer, src=in_edge.src,
                    group=in_edge.process_group, async_op=True)
            if self._pending_req.is_completed():
                # prepare messages for gossip
                mixed_out_msgs = self.mix_out_msg_(out_msg, 1., residual=True)
                msg = next(mixed_out_msgs)
                if self.logger is not None:
                    self.logger.debug('req. completed; sending to {}'.format(out_edge))
                dist.broadcast(tensor=msg, src=out_edge.src,
                               group=out_edge.process_group)
                self._pending_req = None
                completed = True
            else:
                completed = False

        if completed:
            self.refresh_peers_()
            self.clean_msg_buffers_()
            return self.parse_in_msg_buffer(residual=True)

        return out_msg, completed
