import os
import sys
import math
import pprint
import random

# Use for tracking model's loss using Weights & Biases (WandB)
import wandb

import numpy as np

import torch
import torch.distributed as dist
from torch.optim import lr_scheduler

from torchdrug import core, models, tasks, datasets, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import model, cdconv, gvp, dataset, task, protbert


def train_and_validate(cfg, solver, scheduler, working_dir):
    if cfg.train.num_epoch == 0:
        return

    # start a new wandb run to track this script
    wandb.init(project="FusionNetwork", name=cfg.task["class"], config=cfg)

    step = math.ceil(cfg.train.num_epoch / 50)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        # solver.save("model_epoch_%d.pth" % solver.epoch)

        metric = solver.evaluate("valid")
        test_metric = solver.evaluate("test")

        result = metric[cfg.metric]
        test_result = test_metric[cfg.metric]

        wandb.log({f"validation: {cfg.metric}": result})
        wandb.log({f"test: {cfg.metric}": test_result})

        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

        # solver.load("model_epoch_%d.pth" % best_epoch)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)
    return solver


def test(cfg, solver):
    solver.evaluate("valid")
    return solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver, scheduler = util.build_downstream_solver(cfg, dataset)

    train_and_validate(cfg, solver, scheduler, working_dir)
    test(cfg, solver)
