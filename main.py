import argparse
import logging
import multiprocessing as mp
import os
import sys
import time

import hostlist
import torch
import torch.distributed as dist

from cdp_utils import CDP
from data import data_loader, evaluate
from torch.utils.tensorboard import SummaryWriter
from net_utils import create_model, load_optimizer, save_model #add_weight_decay, save_model
from logs_utils import print_training_evolution #, initialize_communications
from torch import _C
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

log = logging.getLogger("distributed_worker")


def get_args_parser():
    parser = argparse.ArgumentParser("Distributed Optimization Script", add_help=False)
    parser.add_argument(
        "--dataset_name",
        default="CIFAR10",
        type=str,
        help="Name of the dataset to train on. We support either one of ['CIFAR10','ImageNet'].",
    )
    parser.add_argument(
        "--model_name",
        default="resnet18",
        type=str,
        help="Name of the model to train. We support either one of ['resnet18', 'resnet50'].",
    )
    parser.add_argument(
        "--normalize_grads",
        default=False,
        action='store_true',
        help="Whether or not to normalize gradients before taking the grad step (for stability issues).",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size to use for each worker.",
    )
    parser.add_argument(
        "--lr",
        default=0.05,
        type=float,
        help="Learning rate of the optimizer.",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        default=5e-4,
        type=float,
        help="Weight decay value of the optimizer.",
    )
    parser.add_argument(
        "--filter_bias_and_bn",
        default=False,
        action='store_true',
        help="Whether or not to prevent weight decay to be applied to biases and batch norm params in the optimizer.",
    )
    parser.add_argument(
        "--split_n_stages",
        default=None,
        type=int,
        help="Number of sub stages to split the model into.",
    )
    parser.add_argument(
        "--n_epoch_if_1_worker",
        default=80,
        type=int,
        help="The number of epochs to perform if there was only one worker. Is used to compute the total number of gradient steps to perform.",
    )
    parser.add_argument(
        "--partition_strategy",
        default='param_size',
        type=str,
        help="How to partition the model in K stages.",
    )
    parser.add_argument(
        "--data_path",
        default='./dataset/imagenet',
        type=str,
        help="Path to the root directory for the data.",
    )
    parser.add_argument(
        "--update_rule",
        default='DP',
        type=str,
        help="Update rule followed. DP|CDP1|CDP2",
    )
    parser.add_argument(
        "--sched_gamma",
        default=0.2,
        type=float,
        help="Scheduler lr reducing gamma",
    )
    parser.add_argument(
        "--resume",
        default=-1,
        type=int,
        help="If resuming",
    )
    parser.add_argument(
        "--test_every_k_epochs",
        default=1,
        type=int,
        help="If evaluate after every epoch",
    )
    parser.add_argument(
        "--memory_record",
        default=False,
        action='store_true',
        help="Record the memory usage for 10 batches.",
    )
    
    return parser


def run(rank, local_rank, world_size, args):
    process_group = dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Initialization of constants and data
    train_loader, image_size, sampler = data_loader(
        train=True,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        with_sampler=True,
        num_replicas=world_size,
        rank=rank,
        data_path=args.data_path
    )
    if not sampler is None:
        sampler.set_epoch(0)
    data_iterator = iter(train_loader)
    n_batch_per_epoch = len(train_loader)

    # Initialize the model, the optimizer, the scheduler
    model, criterion = create_model(args.model_name, args.dataset_name)
    model, criterion = model.to(local_rank), criterion.to(local_rank)

    # Initialize the worker
    cdp_model = CDP(model, rank, local_rank, world_size, args.split_n_stages,
                    args.partition_strategy, image_size)

    optimizer, scheduler = [], []
    for i in range(len(cdp_model.stages_params)):
        if args.weight_decay and args.filter_bias_and_bn:
            params_i = [{"params": cdp_model.stages_params_nodecay[i], "weight_decay": 0.0},
                        {"params": cdp_model.stages_params_decay[i], "weight_decay": args.weight_decay}, ]
        else:
            params_i = cdp_model.stages_params[i]

        # Load an optimizer/scheduler for each stage of the network
        optimizer_i, scheduler_i = load_optimizer(
            params_i,
            args.lr,
            args.momentum,
            args.weight_decay,
            filter_bias_and_bn=args.filter_bias_and_bn,
            sched_gamma=args.sched_gamma,
        )
        optimizer.append(optimizer_i)
        scheduler.append(scheduler_i)

    if args.filter_bias_and_bn:
        filt = "filter"
    else:
        filt = "no_filter"

    cdp_model.weights_to_vec()
    cdp_model.all_reduce(process_group, first_reduce=True)

    epoch = 0

    if args.resume != -1:

        epoch = args.resume
        checkpoint = torch.load(
            f"./checkpoints_cdp/{args.dataset_name}/{args.model_name}_{args.update_rule}_{filt}_{args.resume}.pth",
            map_location="cuda")

        cdp_model.load_state_dict(checkpoint["state_dict"])
        for i in range(len(optimizer)):
            optimizer[i].load_state_dict(checkpoint['optimizer' + str(i)])
            scheduler[i].load_state_dict(checkpoint['scheduler' + str(i)])
        print('Resuming from epoch', str(epoch))

        if epoch % args.test_every_k_epochs == 0:
            test_loader, _, _ = data_loader(train=False, batch_size=args.batch_size, dataset_name=args.dataset_name,
                                            with_sampler=False, data_path=args.data_path)
            loss, correct, len_data, percent = evaluate(cdp_model, test_loader, criterion, rank=local_rank,
                                                        print_message=True)
            print('Test at epoch', str(epoch), 'percent is', str(percent), 'rank', str(rank))

    #### Training loop ####
    t_begin = time.time()
    t_last_epoch = t_begin
    cdp_model.train()
    
    if args.memory_record:
        torch.cuda.memory._record_memory_history(enabled='all')#(enabled=True)#(enabled='all')
    
    count_i = 0
    print("# Start training")
    while epoch < args.n_epoch_if_1_worker:
        count_i += 1
        # get the next batch of data
        try:
            images, labels = next(data_iterator)
        except StopIteration:
            # When the epoch ends, start a new epoch.
            if not sampler is None:
                sampler.set_epoch(epoch)
            data_iterator = iter(train_loader)
            images, labels = next(data_iterator)
            epoch += 1

            # Scheduler step
            for i in range(len(optimizer)):
                scheduler[i].step()

            t_last_epoch = print_training_evolution(log, cdp_model.count_step, n_batch_per_epoch, rank, t_begin,
                                                    t_last_epoch, loss, epoch)

            if rank == 0 and epoch % 10 == 0:
                path = f"./checkpoints_cdp/{args.dataset_name}/{args.model_name}_{args.update_rule}_{filt}_{epoch}.pth"
                save_model(epoch, cdp_model, optimizer, scheduler, args.filter_bias_and_bn, path)

            if epoch % args.test_every_k_epochs == 0 or epoch > args.n_epoch_if_1_worker - 10: # To test imagenet
                test_loader, _, _ = data_loader(train=False, batch_size=args.batch_size, dataset_name=args.dataset_name,
                                                with_sampler=False, data_path=args.data_path)
                loss, correct, len_data, percent = evaluate(cdp_model, test_loader, criterion, rank=local_rank,
                                                            print_message=False)
                print('Test at epoch', str(epoch), 'percent is', str(percent), 'rank', str(rank))

        # distribution of images and labels to all GPUs
        images = images.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)

        # forward pass
        outputs = cdp_model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        for o in optimizer:
            o.zero_grad()

        loss.backward()
        if args.normalize_grads:
            torch.nn.utils.clip_grad_norm_(cdp_model.module.parameters(), 10.0)

        # receive all the gradient of step t
        cdp_model.all_reduce_grad(process_group)
        # update parameters following u(theta_t-1, theta_t)
        cdp_model.update(process_group, optimizer, scheduler, args.update_rule)
        
        if args.memory_record:
            if count_i == 10:
                break
        
    if args.memory_record and rank==0:
        with torch.cuda.device(local_rank):
            #return _C._cuda_memorySnapshot()
            s = _C._cuda_memorySnapshot()
        with open("memory_record2.pickle", "wb") as f:
            print("snap")
            #print(s)
            pickle.dump(s, f)
        print('done')
        #torch.cuda.memory._dump_snapshot("memory_record.pickle")
        
    t_end = time.time()
    #### END OF TRAINNING ####
    cdp_model.all_reduce(process_group)
        
    if rank == 0 and not args.memory_record:

#    if rank == 0:
        path = f"./checkpoints_cdp/{args.dataset_name}/{args.model_name}_{args.update_rule}_{filt}_endavg.pth"
        save_model(epoch, cdp_model, optimizer, scheduler, args.filter_bias_and_bn, path)
        print("FINAL MODEL ACCURACY ")
        test_loader, _, _ = data_loader(train=False, batch_size=args.batch_size, dataset_name=args.dataset_name,
                                        with_sampler=False, data_path=args.data_path)
        loss, correct, len_data, percent = evaluate(cdp_model, test_loader, criterion, rank=local_rank,
                                                    print_message=True)

    torch.distributed.barrier(process_group)


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(
        "Cyclic Data Parallel Script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    # get distributed configuration from Slurm environment
    NODE_ID = os.environ['SLURM_NODEID']
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    n_nodes = len(hostnames)
    # get IDs of reserved GPU
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    # define MASTER_ADD & MASTER_PORT, used to define the distributed communication environment
    master_addr = hostnames[0]
    master_port = 12347 + int(min(gpu_ids))  # to avoid port conflict on the same node
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['MPI4PY_RC_THREADS'] = str(0)  # to avoid problems with MPI in multi-node setting
    # display info
    if rank == 0:
        print(">>> Training on ", n_nodes, " nodes and ", world_size)
        if not os.path.exists(f"./checkpoints_cdp/{args.dataset_name}"):
            os.makedirs(f"./checkpoints_cdp/{args.dataset_name}")
    print("- Process {} corresponds to GPU {} of node {}".format(rank, local_rank, NODE_ID))

    run(rank, local_rank, world_size, args)
