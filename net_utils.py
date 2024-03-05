import torch
import torch.nn as nn
import torchvision


def create_model(model_name="resnet18", dataset_name="CIFAR10"):
    """
    Returns the model corresponding to the given name,
    modified (or not) to handle the given dataset.

    Parameters:
        - model_name (str): the name of the model to load.
                            either one of ['resnet18', ]
        - dataset_name (str): the name of the dataset to use.
                              either one of ['CIFAR10', 'ImageNet']
    Returns:
        - net (nn.Module): the neural net to use.
        - criterion (nn.Module): the criterion to use.
    """
    # sanity check
    if dataset_name not in ["CIFAR10", "ImageNet"]:
        raise ValueError("We only support 'CIFAR10' and 'ImageNet' datasets.")

    # Initialization fit for ImageNet
    modified_conv1 = None
    modified_maxpool = None
    num_classes = 1000
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # adapt to the specificities of CIFAR10
    if dataset_name == "CIFAR10":
        # set the correct number of classes,
        num_classes = 10
        # modify the first conv to handle the image size
        modified_conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        # modify the maxpool layer
        modified_maxpool = nn.Identity()
    # loads the model
    weights = None
    if model_name == "resnet18":
        model = torchvision.models.resnet18(
            weights=weights, num_classes=num_classes
        )
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(
            weights=weights, num_classes=num_classes
        )
    elif model_name == 'vitb16':
        model = torchvision.models.vit_b_16(
            weights=weights, num_classes=num_classes
        )
        modified_conv1,  modifiled_maxpool = None, None

    # update the model if need be
    if modified_conv1 is not None:
        model.conv1 = modified_conv1
    if modified_maxpool is not None:
        model.maxpool = modified_maxpool

    return model, criterion


def add_weight_decay(model, weight_decay, skip_list=()):
    """
    Create 2 sets of parameters: ones to which weight decay should be applied to, and the others (batch norm and bias terms).
    Credit to Ross Wightman https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3

    Parameters:
        - model (Net): the neural net to use.
        - weight_decay (float): value of the weight decay.
        - skip_list (list): the list of parameters names to add to the skip_list.

    Returns:
        - params_list (list of dicts): two sets of parameters, one where no wd is used,
                                       the other where wd is applied.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # in ResNets, len(param.shape) is one only for batch norm weights and bias terms.
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def load_optimizer(
        parameters,
        lr,
        momentum,
        weight_decay,
        filter_bias_and_bn=True,
        sched_gamma=0.2,
):
    """
    Returns the optimizer corresponding to the given name,
    as well as a lr scheduler.
    """
    # loads the model parameters
    if weight_decay and filter_bias_and_bn:
        wd = 0.0
    else:
        wd = weight_decay

    # loads the optimizer
    optimizer = torch.optim.SGD(
        parameters, lr=lr, momentum=momentum, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=sched_gamma)

    return optimizer, scheduler


def save_model(epoch, cdp_model, optimizer, scheduler, filter_bias_and_bn, path=""):
    print("Saving at epoch", epoch)
    checkpoint = {
        'epoch': epoch,
        'state_dict': cdp_model.state_dict(),
    }
    for i in range(len(optimizer)):
        checkpoint['optimizer' + str(i)] = optimizer[i].state_dict()
        checkpoint['scheduler' + str(i)] = scheduler[i].state_dict()

    torch.save(checkpoint, path)
