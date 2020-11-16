

import logging
import os
import time
from multiprocessing import Process
from finrl.model.models import DRLAgent
from finrl.config import config


def train_one(save_path, config, log_file_dir, index, logfile_level, console_level, device):
    """
    train an agent
    :param save_path: the path to save the tensorflow model (.ckpt), could be None
    :param config: the json configuration file
    :param log_file_dir: the directory to save the tensorboard logging file, could be None
    :param index: identifier of this train, which is also the sub directory in the train_package,
    if it is 0. nothing would be saved into the summary file.
    :param logfile_level: logging level of the file
    :param console_level: logging level of the console
    :param device: 0 or 1 to show which gpu to use, if 0, means use cpu instead of gpu
    :return : the Result namedtuple
    """
    if log_file_dir:
        logging.basicConfig(filename=log_file_dir.replace("tensorboard","programlog"),
                            level=logfile_level)
        console = logging.StreamHandler()
        console.setLevel(console_level)
        logging.getLogger().addHandler(console)
    print("training at %s started" % index)
    return TraderTrainer(config, save_path=save_path, device=device).train_net(log_file_dir=log_file_dir, index=index)

def train_all(processes=1, device="cpu"):
    """
    train all the agents in the train_package folders

    :param processes: the number of the processes. If equal to 1, the logging level is debug
                      at file and info at console. If greater than 1, the logging level is
                      info at file and warming at console.
    """
    if processes == 1:
        console_level = logging.INFO
        logfile_level = logging.DEBUG
    else:
        console_level = logging.WARNING
        logfile_level = logging.INFO
    train_dir = "train_package"
    if not os.path.exists("./" + train_dir): #if the directory does not exist, creates one
        os.makedirs("./" + train_dir)
    all_subdir = os.listdir("./" + train_dir)
    all_subdir.sort()
    pool = []
    for dir in all_subdir:
        # train only if the log dir does not exist
        if not str.isdigit(dir):
            return
        # NOTE: logfile is for compatibility reason
        if not (os.path.isdir("./"+train_dir+"/"+dir+"/tensorboard") or os.path.isdir("./"+train_dir+"/"+dir+"/logfile")):
            p = Process(target=train_one, args=(
                "./" + train_dir + "/" + dir + "/netfile",
                load_config(dir),
                "./" + train_dir + "/" + dir + "/tensorboard",
                dir, logfile_level, console_level, device))
            p.start()
            pool.append(p)
        else:
            continue

        # suspend if the processes are too many
        wait = True
        while wait:
            time.sleep(5)
            for p in pool:
                alive = p.is_alive()
                if not alive:
                    pool.remove(p)
            if len(pool)<processes:
                wait = False
    print("All the Tasks are Over")
