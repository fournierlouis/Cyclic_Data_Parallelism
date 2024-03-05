import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from slice_model import split_model_in_k, prepare_flops, equisum_partition


class CDP(nn.Module):
    """
    A wrapper around the model to compute the various update rules
    """

    def __init__(self, model, rank, local_rank, world_size, nb_sub_stages,
                 partition_strategy, image_size):
        super().__init__()

        self.module = model
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.nb_sub_stages = nb_sub_stages
        self.partition_strategy = partition_strategy
        self.image_size = image_size
        # load the size of the stages
        len_stages, self.ids_stage = self.get_stage_sizes()
        self.len_stages = torch.tensor(len_stages).to(self.local_rank)

        count_stage = 0
        id_params = 0
        self.stages_params = [[]]
        self.stages_params_decay = [[]]
        self.stages_params_nodecay = [[]]
        # Prepare stages partition
        for name, param in self.module.named_parameters():
            self.stages_params[-1].append(param)
            # in ResNets, len(param.shape) is one only for batch norm weights and bias terms.
            if len(param.shape) == 1:  # or name in skip_list:
                self.stages_params_decay[-1].append(param)
            else:
                self.stages_params_nodecay[-1].append(param)

            # if the current param is the last belonging to this stage
            if id_params == self.ids_stage[count_stage][-1]:
                count_stage += 1
                if count_stage < self.nb_sub_stages:
                    self.stages_params.append([])
                    self.stages_params_decay.append([])
                    self.stages_params_nodecay.append([])
            id_params += 1

        self.grad_t = None
        self.count_step = 0

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass.
        """
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def get_weights(self, model):
        """
        Given a nn.Module, returns a 1D tensor containing all of its parameters.
        """
        return nn.utils.parameters_to_vector(model.parameters())

    @torch.no_grad()
    def set_weights(self, model, weights):
        """
        Given a 1D tensor containing a nn.Module parameters,
        loads the parameters into the nn.Module.
        """
        nn.utils.vector_to_parameters(weights, model.parameters())

    def weights_to_vec(self):
        # loads the model parameters in a 1D tensor for ease of communication
        params = self.get_weights(self.module)
        params = params.to(self.local_rank)
        self.params = params.share_memory_()
        self.set_weights(self.module, self.params)

    def update(self, process_group, optimizer, scheduler, update_rule):
        self.count_step += 1
        # Update with the new gradients
        if not self.grad_t is None:
            first_pass = False
            self.grad_t_minus_1 = [[gg.clone() for gg in g] for g in self.grad_t]
        else:
            first_pass = True
            self.grad_t_minus_1 = None

        count_stage = 0
        id_params = 0
        self.grad_t = [[]]
        for param in self.module.parameters():
            self.grad_t[-1].append(param.grad.clone())
            # if the current param is the last belonging to this stage
            if id_params == self.ids_stage[count_stage][-1]:
                count_stage += 1
                if count_stage < self.nb_sub_stages:
                    self.grad_t.append([])
            id_params += 1

        dont_update = []

        if update_rule == 'CDP2':
            limit_t = (len(self.stages_params) - 1 - self.rank)
            for l in range(len(self.stages_params)):
                if l >= limit_t:
                    for i, p in enumerate(self.stages_params[l]):
                        p.grad = self.grad_t[l][i].clone()
                else:
                    if first_pass:
                        dont_update.append(l)
                    else:
                        for i, p in enumerate(self.stages_params[l]):
                            p.grad = self.grad_t_minus_1[l][i].clone()

        if update_rule == 'CDP1':
            for l in range(len(self.stages_params)):
                if first_pass:
                    dont_update.append(l)
                else:
                    for i, p in enumerate(self.stages_params[l]):
                        p.grad = self.grad_t_minus_1[l][i].clone()

        if update_rule == 'DP':
            for l in range(len(self.stages_params)):
                for i, p in enumerate(self.stages_params[l]):
                    p.grad = self.grad_t[l][i].clone()

        # Optimizer and scheduler passes
        for l in range(len(self.stages_params)):
            if not l in dont_update:
                optimizer[l].step()

    def all_reduce(self, process_group, first_reduce=False, no_ar_bn=False):
        dist.all_reduce(self.params, group=process_group, op=dist.ReduceOp.SUM)
        self.params.mul_(1 / self.world_size)

        for mod in self.module.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                dist.all_reduce(mod.running_mean, group=process_group, op=dist.ReduceOp.SUM)
                dist.all_reduce(mod.running_var, group=process_group, op=dist.ReduceOp.SUM)
                mod.running_mean.mul_(1 / self.world_size)
                mod.running_var.mul_(1 / self.world_size)

    def all_reduce_grad(self, process_group):
        for param in self.module.parameters():
            dist.all_reduce(param.grad, group=process_group, op=dist.ReduceOp.SUM)
            param.grad.mul_(1 / self.world_size)

    def get_stage_sizes(self):
        if self.nb_sub_stages != -1:
            if self.partition_strategy == 'param_size':
                ids_stage, len_stages = split_model_in_k(self.nb_sub_stages, self.module)
            elif self.partition_strategy == 'nb_stages':
                len_stages = [param.data.numel() for param in self.module.parameters()]
                ids_stage, len_stages = [list(l) for l in np.array_split(range(len(len_stages)), self.nb_sub_stages)], [
                    np.sum(l) for l in np.array_split(len_stages, self.nb_sub_stages)]
            elif self.partition_strategy == 'flops':
                flops = FlopCountAnalysis(self.module,
                                          torch.randn(1, 3, self.image_size, self.image_size).to(self.local_rank))
                dict_flops = dict(flops.by_module())
                list_flops = prepare_flops(self.module, dict_flops)
                ids_stage, len_stages = equisum_partition(list_flops, self.nb_sub_stages)
                print("Partition:", ids_stage)
                print("Percentage of FLOPS:", np.array(len_stages) / np.sum(len_stages))
            else:
                raise ('Not a valid partition strategy')
        else:
            # each parameter in the NN is its own stage
            len_stages = [param.data.numel() for param in self.module.parameters()]
            ids_stage = [[i] for i in range(len(len_stages))]

        # re-write the true nb of stages
        self.nb_sub_stages = len(len_stages)
        return len_stages, ids_stage
