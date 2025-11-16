# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import signal
import socket
import sys
import json
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim

from torch.cuda.amp import GradScaler
from pytorch_lightning.lite import LightningLite

from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline

from cotracker.utils.visualizer import Visualizer
from cotracker.models.core.model_utils import get_uniformly_sampled_pts
from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
from cotracker.models.core.cotracker.losses import (
    sequence_loss,
    sequence_BCE_loss,
    sequence_prob_loss,
)
from cotracker.utils.train_utils import (
    Logger,
    get_eval_dataloader,
    get_train_dataset,
    sig_handler,
    term_handler,
    run_test_eval,
)


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    mlp_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and "corr_mlp" in name
    )
    print(f"Total number of MlP parameters: {mlp_params}")

    mlp_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and "cmdtop" in name
    )
    print(f"Total number of cmdtop parameters: {mlp_params}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="cos",
    )
    return optimizer, scheduler


def forward_batch(batch, model, args):
    video = batch.video
    trajs_g = batch.trajectory
    vis_g = batch.visibility
    valids = batch.valid

    B, T, C, H, W = video.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = video.device

    # Queries will be the ground truth coordinates at the first frame where it's visible and valid.
    vis_val = vis_g * valids  # [B, T, N]
    queried_frames = vis_val.max(dim=1).indices[..., None]  # [B, N, 1]
    queried_coords = trajs_g.gather(1, queried_frames[:, None].repeat(1, 1, 1, 2)).squeeze(1)  # [B, N, 2]
    queries = torch.cat([queried_frames, queried_coords], dim=2)  # [B, N, 3]

    assert B == 1

    if (
        torch.isnan(queries).any()
        or torch.isnan(trajs_g).any()
        or queries.abs().max() > 1500
    ):
        print("failed_sample")
        print("queries time", queries[..., 0])
        print("queries ", queries[..., 1:])
        queries = torch.ones_like(queries).to(queries.device).float()
        print("new queries", queries)
        valids = torch.zeros_like(valids).to(valids.device).float()
        print("new valids", valids)

    model_output = model(
        video=video, queries=queries[..., :3], iters=args.train_iters, is_train=True
    )

    tracks, visibility, confidence, train_data = model_output
    coord_predictions, vis_predictions, confidence_predicitons, valid_mask = train_data

    vis_gts = []
    invis_gts = []
    traj_gts = []
    valids_gts = []

    if args.offline_model:
        S = T
        seq_len = (S // 2) + 1
    else:
        S = args.sliding_window_len
        seq_len = T

    for ind in range(0, seq_len - S // 2, S // 2):
        vis_gts.append(vis_g[:, ind : ind + S])
        invis_gts.append(1 - vis_g[:, ind : ind + S])
        traj_gts.append(trajs_g[:, ind : ind + S, :, :2])
        val = valids[:, ind : ind + S]
        if not args.offline_model:
            val = val * valid_mask[:, ind : ind + S]
        valids_gts.append(val)

    seq_loss_visible = sequence_loss(
        coord_predictions,
        traj_gts,
        valids_gts,
        vis=vis_gts,
        gamma=0.8,
        add_huber_loss=args.add_huber_loss,
        loss_only_for_visible=True,
    )
    confidence_loss = sequence_prob_loss(
        coord_predictions, confidence_predicitons, traj_gts, vis_gts
    )
    vis_loss = sequence_BCE_loss(vis_predictions, vis_gts)

    output = {"flow": {"predictions": tracks[0].detach()}}
    output["flow"]["loss"] = seq_loss_visible.mean() * 0.05
    output["flow"]["queries"] = queries.clone()

    if not args.train_only_on_visible:
        seq_loss_invisible = sequence_loss(
            coord_predictions,
            traj_gts,
            valids_gts,
            vis=invis_gts,
            gamma=0.8,
            add_huber_loss=False,
            loss_only_for_visible=True,
        )
        output["flow_invisible"] = {"loss": seq_loss_invisible.mean() * 0.01}
    output["visibility"] = {
        "loss": vis_loss.mean(),
        "predictions": visibility[0].detach(),
    }
    output["confidence"] = {
        "loss": confidence_loss.mean(),
    }
    return output


class Lite(LightningLite):
    def run(self, args):
        def seed_everything(seed: int):
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(42)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed + worker_id)
            random.seed(worker_seed + worker_id)

        g = torch.Generator()
        g.manual_seed(42)
        if self.global_rank == 0:
            eval_dataloaders = []
            for ds_name in args.eval_datasets:
                eval_dataloaders.append(
                    (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                )
            if not args.debug:
                final_dataloaders = [dl for dl in eval_dataloaders]

                # ds_name = "dynamic_replica"
                # final_dataloaders.append(
                #     (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                # )

                # ds_name = "tapvid_robotap"
                # final_dataloaders.append(
                #     (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                # )

                # ds_name = "tapvid_kinetics_first"
                # final_dataloaders.append(
                #     (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                # )

            evaluator = Evaluator(args.ckpt_path)

            visualizer = Visualizer(
                save_dir=args.ckpt_path,
                pad_value=180,
                fps=1,
                show_first_frame=0,
                tracks_leave_trace=0,
            )

        if args.model_name == "cotracker_three":
            if args.offline_model:
                model = CoTrackerThreeOffline(
                    stride=args.model_stride,
                    corr_radius=args.corr_radius,
                    corr_levels=args.corr_levels,
                    window_len=args.sliding_window_len,
                    num_virtual_tracks=args.num_virtual_tracks,
                    model_resolution=args.crop_size,
                    linear_layer_for_vis_conf=args.linear_layer_for_vis_conf,
                )
            else:
                model = CoTrackerThreeOnline(
                    stride=args.model_stride,
                    corr_radius=args.corr_radius,
                    corr_levels=args.corr_levels,
                    window_len=args.sliding_window_len,
                    num_virtual_tracks=args.num_virtual_tracks,
                    model_resolution=args.crop_size,
                    linear_layer_for_vis_conf=args.linear_layer_for_vis_conf,
                )
        else:
            raise ValueError(f"Model {args.model_name} doesn't exist")

        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        model.cuda()

        train_dataset = get_train_dataset(args)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
            collate_fn=collate_fn_train,
            drop_last=True,
        )
        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)
        print("LEN TRAIN LOADER", len(train_loader))
        optimizer, scheduler = fetch_optimizer(args, model)

        total_steps = 0
        if self.global_rank == 0:
            logger = Logger(model, scheduler, ckpt_path=args.ckpt_path)

        folder_ckpts = [
            f
            for f in os.listdir(args.ckpt_path)
            if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f
        ]
        if len(folder_ckpts) > 0:
            ckpt_path = sorted(folder_ckpts)[-1]
            ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
            logging.info(f"Loading checkpoint {ckpt_path}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")

        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(
                ".pt"
            )
            logging.info("Loading checkpoint...")

            strict = False
            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if "time_emb" not in k and "pos_emb" not in k
            }
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict, strict=strict)

            logging.info(f"Done loading checkpoint")

        print(f"fnet param count (trainable): {sum(p.numel() for p in model.fnet.parameters() if p.requires_grad)}")
        print(f"fnet param count (total): {sum(p.numel() for p in model.fnet.parameters())}")
        print(f"updateformer param count (trainable): {sum(p.numel() for p in model.updateformer.parameters() if p.requires_grad)}")
        print(f"updateformer param count (total): {sum(p.numel() for p in model.updateformer.parameters())}")
        print(f"CoTracker param count (trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"CoTracker param count (total): {sum(p.numel() for p in model.parameters())}")

        model, optimizer = self.setup(model, optimizer, move_to_device=False)
        model.train()

        save_freq = args.save_freq
        scaler = GradScaler(enabled=False)

        should_keep_training = True
        global_batch_num = 0
        epoch = -1

        while should_keep_training:
            epoch += 1
            for i_batch, batch in enumerate(tqdm(train_loader)):
                batch, gotit = batch
                if not all(gotit):
                    print("batch is None")
                    continue

                dataclass_to_cuda_(batch)

                optimizer.zero_grad(set_to_none=True)

                assert model.training

                output = forward_batch(batch, model, args)

                loss = 0
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]

                if self.global_rank == 0:
                    for k, v in output.items():
                        if "loss" in v:
                            logger.writer.add_scalar(
                                f"live_{k}_loss", v["loss"].item(), total_steps
                            )
                        if "metrics" in v:
                            logger.push(v["metrics"], k)
                    if total_steps % save_freq == save_freq - 1:
                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=batch.trajectory.clone()[..., :2],
                            visibility=batch.visibility.clone(),
                            filename="train_gt_traj_0",
                            writer=logger.writer,
                            step=total_steps,
                        )

                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=output["flow"]["predictions"][None],
                            visibility=output["visibility"]["predictions"][None] > 0.8,
                            filename="train_pred_traj_0",
                            writer=logger.writer,
                            step=total_steps,
                        )

                    if len(output) > 1:
                        logger.writer.add_scalar(
                            f"live_total_loss", loss.item(), total_steps
                        )
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    global_batch_num += 1

                self.barrier()
                self.backward(scaler.scale(loss))

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                total_steps += 1
                if self.global_rank == 0:
                    if total_steps % args.save_every_n_steps == 0:
                        ckpt_iter = "0" * (6 - len(str(total_steps))) + str(
                            total_steps
                        )
                        save_path = Path(
                            f"{args.ckpt_path}/model_{args.model_name}_{ckpt_iter}.pth"
                        )

                        save_dict = {
                            "model": model.module.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "total_steps": total_steps,
                        }

                        logging.info(f"Saving file {save_path}")
                        self.save(save_dict, save_path)

                    if total_steps % args.evaluate_every_n_steps == 0 or (
                        args.validate_at_start and total_steps == 0
                    ):
                        run_test_eval(
                            evaluator,
                            model,
                            eval_dataloaders,
                            logger.writer,
                            total_steps,
                            query_random=False,
                        )
                        model.train()
                        torch.cuda.empty_cache()

                self.barrier()
                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

        if self.global_rank == 0:
            print("FINISHED TRAINING")

            PATH = f"{args.ckpt_path}/{args.model_name}_final.pth"
            torch.save(model.module.module.state_dict(), PATH)
            run_test_eval(
                evaluator,
                model,
                final_dataloaders,
                logger.writer,
                total_steps,
                query_random=False,
            )
            logger.close()


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="cotracker_three", help="model name")
    parser.add_argument("--restore_ckpt", help="path to restore a checkpoint")
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size used during training."
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--num_workers", type=int, default=10, help="number of dataloader workers"
    )

    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="max learning rate.")
    parser.add_argument(
        "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200000, help="length of training schedule."
    )
    parser.add_argument(
        "--evaluate_every_n_steps",
        type=int,
        default=2000,
        help="evaluate during training after every n steps, after every 2000 steps by default",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=2000,
        help="save checkpoints during training after every n steps, after every 2000 steps by default",
    )
    parser.add_argument(
        "--validate_at_start",
        action="store_true",
        help="whether to run evaluation before training starts",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="frequency of trajectory visualization during training",
    )
    parser.add_argument(
        "--traj_per_sample",
        type=int,
        default=768,
        help="the number of trajectories to sample for training",
    )
    parser.add_argument(
        "--dataset_root", type=str, help="path lo all the datasets (train and eval)"
    )

    parser.add_argument(
        "--train_iters",
        type=int,
        default=4,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument(
        "--sequence_len", type=int, default=8, help="train sequence length"
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["tapvid_davis_first"],
        help="what datasets to use for evaluation",
    )
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        default=["kubric"],
        help="what datasets to use for evaluation",
    )
    parser.add_argument(
        "--num_virtual_tracks",
        type=int,
        default=None,
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--dont_use_augs",
        action="store_true",
        help="don't apply augmentations during training",
    )
    parser.add_argument(
        "--offline_model",
        action="store_true",
        help="only sample trajectories with points visible on the first frame",
    )
    parser.add_argument(
        "--sliding_window_len",
        type=int,
        default=16,
        help="length of the CoTracker sliding window",
    )
    parser.add_argument(
        "--model_stride",
        type=int,
        default=4,
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--corr_radius",
        type=int,
        default=3,
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--corr_levels",
        type=int,
        default=4,
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=[384, 512],
        help="crop videos to this resolution during training",
    )
    parser.add_argument(
        "--add_huber_loss",
        action="store_true",
        help="only sample trajectories with points visible on the first frame",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="only sample trajectories with points visible on the first frame",
    )
    parser.add_argument(
        "--linear_layer_for_vis_conf",
        action="store_true",
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--train_only_on_visible",
        action="store_true",
        help="stride of the CoTracker feature network",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=False),
        devices="auto",
        accelerator="gpu",
        precision="bf16" if args.mixed_precision else 32,
        num_nodes=args.num_nodes,
    ).run(args)
