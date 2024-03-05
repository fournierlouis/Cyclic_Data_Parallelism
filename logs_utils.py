import datetime
import random
import time


def create_id_run():
    """
    Create a unique id for the current run using the date
    """
    time_now = datetime.datetime.now()
    id_run = "_".join(
        [
            str(time_part)
            for time_part in [
            time_now.year,
            time_now.month,
            time_now.day,
            time_now.hour,
            time_now.minute,
            time_now.second,
        ]
        ]
    )
    # to differentiate runs launched exactly at the same time on the cluster.
    random_number = random.randint(0, 100)
    id_run += "_" + str(random_number)
    return id_run


def print_training_evolution(log, nb_grad_local, n_batch_per_epoch, rank, t_beg, t_last_epoch, loss, epoch):
    delta_t = time.time() - t_beg
    log.info(
        " Worker {}. Epoch {} in {:.2f} s. Total time: {} min {:.2f} s. # grad : {}. loss {}".format(
            rank,
            epoch,
            time.time() - t_last_epoch,
            int(delta_t // 60),
            delta_t % 60,
            nb_grad_local,
            float(loss.detach().cpu()),
        )
    )
    t_last_epoch = time.time()
    return t_last_epoch
