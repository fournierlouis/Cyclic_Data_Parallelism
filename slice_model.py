import numpy as np


def prepare_flops(model, flops):
    names = [p[0] for p in model.named_parameters()]
    list_flops = []
    for n in names:
        n = n.replace('.weight', '')
        if n != n.replace('.bias', ''):
            list_flops.append(np.sqrt(flops[n.replace('.bias', '')]))
        else:
            list_flops.append(flops[n])

    return (list_flops)


def equisum_partition(arr, p):
    arr = np.array(arr)

    ac = arr.cumsum()

    # sum of the entire array
    partsum = ac[-1] // p

    # generates the cumulative sums of each part
    cumpartsums = np.array(range(1, p)) * partsum

    # finds the indices where the cumulative sums are sandwiched
    inds = np.searchsorted(ac, cumpartsums)

    # split into approximately equal-sum arrays
    parts = np.split(arr, inds)

    k = 0
    ids = []
    for p in parts:
        ids.append(list(range(k, k + len(p))))
        k += len(p)

    parts = [np.sum(p) for p in parts]
    return ids, parts


def split_model_in_k(k, model, list_weights=None):
    """
    Heuristic running in O(number of weights) to slice a model in k contiguous subsets of weights that countain roughly the same number of parameters.
    The problem is NP-hard ( https://www.wikiwand.com/en/Multiway_number_partitioning ).
    We begin by initializing k slices by attributing the k largest parameters of the model to a different slice.
    Then, we progressively grow the smallest slice, one parameter at a time, until all slices are "completed".
    A slice is complete when it cannot grow neither to the left or right.
    We return the ids of the weights contained in each slice, as well as there total number of parameters.
    """
    if list_weights is None:
        list_weights = [param.data.numel() for param in model.parameters()]
    n_parts = len(list_weights)
    is_weight_attributed = [False] * n_parts
    sorted_weights_ids = np.argsort(list_weights)
    # case where the user asks more division than possible
    k = min([n_parts, k])
    # we initialize the slices centroids with the k largest weights
    ids_init = sorted_weights_ids[-k:]
    # sort the list so that in the end, slice 0 begin at id 0, and slice -1 finish at id -1
    ids_init.sort()
    ids_to_slice_k = [[ids_init[i]] for i in range(k)]
    weights_to_slice_k = [list_weights[ids_init[i]] for i in range(k)]
    # init the boolean list
    for i in range(k):
        is_weight_attributed[ids_init[i]] = True
    # init 2 lists that keeps track of all "not completed" slice.
    # a slice is said to be completed when it cannot grow both on the left or the right,
    # i.e it touches a bordure or an other slice on both sides.
    ids_not_completed_slice = list(range(k))
    weights_not_completed_slice = [list_weights[ids_init[i]] for i in range(k)]
    # while there are slices to be completed
    while len(ids_not_completed_slice) > 0:
        # gather the id of the not completed slice with the least weight
        id_in_list = np.argmin(weights_not_completed_slice)
        id_slice = ids_not_completed_slice[id_in_list]
        # check whether or not we can grow the slice
        is_completed, available_ids_to_grow = slice_is_completed(id_slice, ids_to_slice_k, is_weight_attributed)
        # if the slice is completed, we remove it from the list of uncompleted ones
        if is_completed:
            ids_not_completed_slice.pop(id_in_list)
            weights_not_completed_slice.pop(id_in_list)
        # else, we grow it by attaching to it the largest weight available its bordures touches
        else:
            id_weight_to_attach = available_ids_to_grow[
                np.argmax([list_weights[id_available] for id_available in available_ids_to_grow])]
            # attach the weight to the slice
            weights_not_completed_slice[id_in_list] += list_weights[id_weight_to_attach]
            weights_to_slice_k[id_in_list] += list_weights[id_weight_to_attach]
            is_weight_attributed[id_weight_to_attach] = True
            ids_to_slice_k[id_slice].append(id_weight_to_attach)
            # sort the list, so that the right and left end correspond to the limits of the slice
            ids_to_slice_k[id_slice].sort()

    return ids_to_slice_k, weights_to_slice_k


def slice_is_completed(id_slice, ids_to_slice_k, is_weight_attributed):
    # gather the ends of the slice
    slice_ids = ids_to_slice_k[id_slice]
    left_end, right_end = slice_ids[0], slice_ids[-1]
    # count, if its = 2, it means both ends are blocked
    test_both_dead_end = 0
    available_ids_to_grow = []
    # check if we are at the beginning of the list
    if left_end == 0:
        test_both_dead_end += 1
    else:
        # if the weight to the left is already in a slice
        if is_weight_attributed[left_end - 1]:
            test_both_dead_end += 1
        else:
            available_ids_to_grow.append(left_end - 1)
    # check the right end
    if right_end == len(is_weight_attributed) - 1:
        test_both_dead_end += 1
    else:
        # if the weight to the right is already in a slice
        if is_weight_attributed[right_end + 1]:
            test_both_dead_end += 1
        else:
            available_ids_to_grow.append(right_end + 1)
    is_completed = test_both_dead_end == 2

    return is_completed, available_ids_to_grow
