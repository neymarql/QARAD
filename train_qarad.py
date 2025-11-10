import warnings
import argparse
import torch
from datasets.data_builder import build_dataloader
from easydict import EasyDict
import yaml
import os
from utils.misc_helper import get_current_time,create_logger,set_seed,AverageMeter,save_checkpoint,summary_model,EarlyStopping

from models.model_helper import ModelHelper
from utils.optimizer_helper import get_optimizer
from utils.criterion_helper import build_criterion
from utils.eval_helper import performances,log_metrics
import logging
import numpy as np
import pprint
from utils.dist_helper import setup_distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.categories import Categories
import torch.multiprocessing as mp
import csv
import time
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="train QARAD")
parser.add_argument("--config", default="experiments/{}/qarad.yaml")
parser.add_argument("--dataset", default="MVTec-AD",choices=['MVTec-AD','VisA','MPDD','BTAD'])
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--class_name", default="bottle",choices=[
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
        "bracket_black",
        "bracket_brown",
        "bracket_white",
        "connector",
        "metal_plate",
        "tubes",
         "01",
         "02",
         "03",
        ] )


def update_config(config,args):
    layers=[]
    for block in config.structure:
        layers.extend([layer.idx for layer in block.layers])
    layers=list(set(layers))
    layers.sort()

    config.net[0].kwargs['outlayers']=layers

    config.net[1].kwargs=config.net[1].get('kwargs',{})
    config.net[1].kwargs['structure']=config.structure

    config.dataset.train.meta_file = config.dataset.train.meta_file.replace("{}", args.class_name)
    config.dataset.test.meta_file = config.dataset.test.meta_file.replace("{}", args.class_name)
    if hasattr(config.dataset.train, 'sdas_dir'):
        config.dataset.train.sdas_dir = config.dataset.train.sdas_dir.replace("{}", args.class_name)
    if hasattr(config.dataset, 'category'):
        config.dataset.category = config.dataset.category.replace("{}", args.class_name)
    return config


def main():
    args = parser.parse_args()

    class_name_list=Categories[args.dataset]
    assert args.class_name in class_name_list

    args.config=args.config.format(args.dataset)

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    rank, world_size = setup_distributed()

    set_seed(config.random_seed)
    config = update_config(config,args)

    config.exp_path = os.path.dirname(args.config)
    config.checkpoints_path = os.path.join(config.exp_path, config.saver.checkpoints_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    patience = config.trainer.early_stop.patience
    delta    = config.trainer.early_stop.delta

    if rank==0:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()

        logger = create_logger(
            "qarad_logger_{}".format(args.class_name), config.log_path + "/qarad_{}_{}.log".format(args.class_name,current_time)
        )

        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
        logger.info("class name is : {}".format(args.class_name))


    train_loader, val_loader = build_dataloader(config.dataset, distributed=True)

    local_rank = int(os.environ["LOCAL_RANK"])

    model = ModelHelper(config.net)
    model.cuda()

    if rank == 0:
        summary_model(model, logger)

    model.afs.init_idxs(model, train_loader, distributed=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    layers = []
    for module in config.net:
        layers.append(module["name"])

    frozen_layers = model.module.frozen_layers
    active_layers = [ layer for layer in layers if layer not in frozen_layers]

    if rank==0:
        logger.info("layers: {}".format(layers))
        logger.info("frozen layers: {}".format(frozen_layers))
        logger.info("active layers: {}".format(active_layers))

    parameters = [
        {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)

    if not hasattr(config, "evaluator"):
        config.evaluator = EasyDict()
    config.evaluator.setdefault("key_metric", "mean")
    config.evaluator.setdefault("metrics", {"auc": []})

    key_metric = config.evaluator["key_metric"]

    best_metric = 0
    early_stop_counter = 0
    should_stop = False
    checkpoint_dir = os.path.join(config.checkpoints_path, args.class_name)


    criterion = build_criterion(config.criterion)
    early_stopper = EarlyStopping(
        patience = config.trainer.early_stop.patience,
        delta    = config.trainer.early_stop.delta,
        larger_better = True,
        rank     = rank
    )
    for epoch in range(0, config.trainer.max_epoch):
        if should_stop:
            break
        last_iter = epoch * len(train_loader)
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(
            config,
            train_loader,
            model,
            optimizer,
            epoch,
            last_iter,
            criterion,
            args.class_name
        )

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:

            ret_metrics = validate(config,val_loader, model, epoch + 1,args.class_name)

            if rank==0:
                ret_key_metric = np.mean([ret_metrics[key] for key in ret_metrics if key.find(key_metric)!=-1])


                is_best = early_stopper.step(ret_key_metric)
                best_metric = early_stopper.best

                if is_best:
                    early_stop_counter = 0
                    best_record = {key.replace("mean",'best') :ret_metrics[key] for key in ret_metrics if key.find("mean")!=-1}
                else:
                    early_stop_counter += 1

                ret_metrics.update(best_record)
                log_metrics(ret_metrics, config.evaluator.metrics, "qarad_logger_{}".format(args.class_name))
                if is_best:
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "arch": config.net,
                            "state_dict": model.module.state_dict(),
                            "best_metric": best_metric,
                            "optimizer": optimizer.state_dict(),
                            "early_stop_counter": early_stop_counter,
                        },
                        config,
                        args.class_name,
                    )
                stop_tensor = torch.tensor(int(early_stopper.stop), device='cuda')
            else:
                stop_tensor = torch.tensor(0, device='cuda')
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                if rank == 0:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}. "
                                f"Best {key_metric}: {best_metric:.5f}")
                break
        dist.barrier()

def train_one_epoch(
    config,
    train_loader,
    model,
    optimizer,
    epoch,
    start_iter,
    criterion,
    class_name
):

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logger = logging.getLogger("qarad_logger_{}".format(class_name))


    losses = AverageMeter(config.trainer.print_freq_step)
    model.train()
    start_time = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        outputs = model(input, train=True)
        loss = []
        loss_terms = {}
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            term = weight * criterion_loss(outputs)
            loss_terms[name] = term.mean()
            loss.append(term)
        loss = torch.sum(torch.stack(loss))
        if 'weight' in input:
            loss = loss * input['weight'].to(loss.device)
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        reduced_loss = reduced_loss.mean()
        losses.update(reduced_loss.item())
        reduced_terms = {}
        for name, val in loss_terms.items():
            r = val.clone()
            dist.all_reduce(r)
            r = r / world_size
            reduced_terms[name] = r.item()
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if rank==0 and (curr_step + 1) % config.trainer.print_freq_step == 0:
            msg = (
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
            ).format(
                epoch + 1,
                config.trainer.max_epoch,
                curr_step + 1,
                len(train_loader) * config.trainer.max_epoch,
                loss=losses,
            )
            if reduced_terms:
                term_str = ": ".join([f"{k} {v:.5f}" for k, v in reduced_terms.items()])
                msg = msg + term_str
            logger.info(msg)


def validate(config,val_loader, model, epoch, class_name):

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model.eval()
    losses = []

    if rank == 0:
        logger = logging.getLogger("qarad_logger_{}".format(class_name))

    criterion = build_criterion(config.criterion)

    preds = []
    masks = []

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            outputs = model(input, train=False)

            preds.append(outputs["anomaly_score"])
            masks.append(outputs["mask"])

            loss = []
            for _, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss.append(weight * criterion_loss(outputs))

            loss = torch.sum(torch.stack(loss))
            losses.append(loss.item())

    preds = torch.cat(preds, dim=0)
    masks = torch.cat(masks, dim=0)

    if world_size > 1:
        preds_list = [torch.zeros_like(preds) for _ in range(world_size)]
        masks_list = [torch.zeros_like(masks) for _ in range(world_size)]
        dist.all_gather(preds_list, preds)
        dist.all_gather(masks_list, masks)
        preds = torch.cat(preds_list, dim=0)[: len(val_loader.dataset)]
        masks = torch.cat(masks_list, dim=0)[: len(val_loader.dataset)]

    loss_tensor = torch.tensor(np.mean(losses), device=preds.device)
    if world_size > 1:
        dist.all_reduce(loss_tensor)
        loss_tensor /= world_size

    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()

    preds = np.squeeze(preds, axis=1)
    masks = np.squeeze(masks, axis=1)

    if rank == 0:
        logger.info(" * Loss {:.5f}".format(loss_tensor.item()))
        ret_metrics = performances(class_name, preds, masks, config.evaluator.metrics)
    else:
        ret_metrics = None

    model.train()
    return ret_metrics
    


if __name__ == "__main__":
    main()
