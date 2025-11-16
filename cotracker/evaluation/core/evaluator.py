# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import os
from typing import Optional, Dict
import torch
from tqdm import tqdm
import numpy as np
import time
import torch.linalg as LA
from einops import repeat

from torch.utils.tensorboard import SummaryWriter
from cotracker.datasets.utils import dataclass_to_cuda_
from cotracker.utils.visualizer import Visualizer
from cotracker.models.core.model_utils import reduce_masked_mean, reduce_masked_median
from cotracker.evaluation.core.eval_utils import compute_tapvid_metrics
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline
import logging


class Evaluator:
    """
    A class defining the CoTracker evaluator.
    """

    def __init__(self, exp_dir) -> None:
        # Visualization
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.visualization_filepaths = defaultdict(lambda: defaultdict(list))
        self.visualize_dir = os.path.join(exp_dir, "visualisations")

    def compute_metrics(self, metrics, sample, pred_trajectory, dataset_name):
        if isinstance(pred_trajectory, tuple):
            pred_trajectory, pred_visibility = pred_trajectory
        else:
            pred_visibility = None
        if "tapvid" in dataset_name:
            B, T, N, D = sample.trajectory.shape
            # traj = sample.trajectory.clone()
            # thr = 0.6

            # if pred_visibility is None:
            #     logging.warning("visibility is NONE")
            #     pred_visibility = torch.zeros_like(sample.visibility)

            # if not pred_visibility.dtype == torch.bool:
            #     pred_visibility = pred_visibility > thr

            # query_points = sample.query_points.clone().cpu().numpy()

            # gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
            # gt_occluded = (
            #     torch.logical_not(sample.visibility.clone().permute(0, 2, 1))
            #     .cpu()
            #     .numpy()
            # )

            # pred_occluded = (
            #     torch.logical_not(pred_visibility.clone().permute(0, 2, 1))
            #     .cpu()
            #     .numpy()
            # )
            # pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()

            # out_metrics = compute_tapvid_metrics(
            #     query_points,
            #     gt_occluded,
            #     gt_tracks,
            #     pred_occluded,
            #     pred_tracks,
            #     query_mode="strided" if "strided" in dataset_name else "first",
            # )

            # metrics[sample.seq_name[0]] = out_metrics
            # for metric_name in out_metrics.keys():
            #     if "avg" not in metrics:
            #         metrics["avg"] = {}
            #     metrics["avg"][metric_name] = np.mean(
            #         [v[metric_name] for k, v in metrics.items() if k != "avg"]
            #     )

            # logging.info(f"Metrics: {out_metrics}")
            # logging.info(f"avg: {metrics['avg']}")
            # print("metrics", out_metrics)
            # print("avg", metrics["avg"])

            # UNCOMMENT
            H, W = sample.video.shape[-2:]
            device = sample.video.device
            out_metrics = {}
            d_vis_sum = d_occ_sum = d_sum_all = 0.0
            thrs = [1, 2, 4, 8, 16]
            sx_ = (W - 1) / 255.0
            sy_ = (H - 1) / 255.0
            sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
            sc_pt = torch.from_numpy(sc_py).float().to(device)
            __, first_visible_inds = torch.max(sample.visibility, dim=1)

            frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(
                B, 1, N
            )
            start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))

            for thr in thrs:
                d_ = (
                    torch.norm(
                        pred_trajectory[..., :2] / sc_pt
                        - sample.trajectory[..., :2] / sc_pt,
                        dim=-1,
                    )
                    < thr
                ).float()  # B,S-1,N
                d_occ = (
                    reduce_masked_mean(
                        d_,
                        (1 - sample.visibility.float())
                        * start_tracking_mask
                        * sample.valid,
                    ).item()
                    * 100.0
                )
                d_occ_sum += d_occ
                out_metrics[f"accuracy_occ_{thr}"] = d_occ

                d_vis = (
                    reduce_masked_mean(
                        d_, sample.visibility * start_tracking_mask * sample.valid
                    ).item()
                    * 100.0
                )
                d_vis_sum += d_vis
                out_metrics[f"accuracy_vis_{thr}"] = d_vis

                d_all = (
                    reduce_masked_mean(d_, start_tracking_mask * sample.valid).item()
                    * 100.0
                )
                d_sum_all += d_all
                out_metrics[f"accuracy_{thr}"] = d_all

            d_occ_avg = d_occ_sum / len(thrs)
            d_vis_avg = d_vis_sum / len(thrs)
            d_all_avg = d_sum_all / len(thrs)

            sur_thr = 16
            dists = torch.norm(
                pred_trajectory[..., :2] / sc_pt - sample.trajectory[..., :2] / sc_pt,
                dim=-1,
            )  # B,S,N
            dist_ok = 1 - (dists > sur_thr).float() * sample.visibility  # B,S,N
            survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
            out_metrics["survival"] = torch.mean(survival).item() * 100.0
            out_metrics["accuracy_occ"] = d_occ_avg
            out_metrics["accuracy_vis"] = d_vis_avg
            out_metrics["accuracy"] = d_all_avg

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            print("metrics", out_metrics)
            print("avg", metrics["avg"])
        elif dataset_name == "dynamic_replica" or dataset_name == "pointodyssey":
            *_, N, _ = sample.trajectory.shape
            B, T, N = sample.visibility.shape
            H, W = sample.video.shape[-2:]
            device = sample.video.device

            out_metrics = {}

            d_vis_sum = d_occ_sum = d_sum_all = 0.0
            thrs = [1, 2, 4, 8, 16]
            sx_ = (W - 1) / 255.0
            sy_ = (H - 1) / 255.0
            sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
            sc_pt = torch.from_numpy(sc_py).float().to(device)
            __, first_visible_inds = torch.max(sample.visibility, dim=1)

            frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(
                B, 1, N
            )
            start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))

            for thr in thrs:
                d_ = (
                    torch.norm(
                        pred_trajectory[..., :2] / sc_pt
                        - sample.trajectory[..., :2] / sc_pt,
                        dim=-1,
                    )
                    < thr
                ).float()  # B,S-1,N
                d_occ = (
                    reduce_masked_mean(
                        d_, (1 - sample.visibility) * start_tracking_mask * sample.valid
                    ).item()
                    * 100.0
                )
                d_occ_sum += d_occ
                out_metrics[f"accuracy_occ_{thr}"] = d_occ

                d_vis = (
                    reduce_masked_mean(
                        d_, sample.visibility * start_tracking_mask * sample.valid
                    ).item()
                    * 100.0
                )
                d_vis_sum += d_vis
                out_metrics[f"accuracy_vis_{thr}"] = d_vis

                d_all = (
                    reduce_masked_mean(d_, start_tracking_mask * sample.valid).item()
                    * 100.0
                )
                d_sum_all += d_all
                out_metrics[f"accuracy_{thr}"] = d_all

            d_occ_avg = d_occ_sum / len(thrs)
            d_vis_avg = d_vis_sum / len(thrs)
            d_all_avg = d_sum_all / len(thrs)

            sur_thr = 16
            dists = torch.norm(
                pred_trajectory[..., :2] / sc_pt - sample.trajectory[..., :2] / sc_pt,
                dim=-1,
            )  # B,S,N
            dist_ok = (
                1 - (dists > sur_thr).float() * sample.visibility * sample.valid
            )  # B,S,N
            survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
            out_metrics["survival"] = torch.mean(survival).item() * 100.0

            out_metrics["accuracy_occ"] = d_occ_avg
            out_metrics["accuracy_vis"] = d_vis_avg
            out_metrics["accuracy"] = d_all_avg

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            print("metrics", out_metrics)
            print("avg", metrics["avg"])

    @torch.no_grad()
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        visualize_every: int = 50,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
        num_samples: int = 1,
        worst_of_n: bool = False,
        use_oracle: bool = True,
    ):
        metrics = {}

        vis = Visualizer(
            save_dir=self.exp_dir,
            fps=12,
        )

        for ind, sample in enumerate(tqdm(test_dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue
            if torch.cuda.is_available():
                dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            if (
                not train_mode
                and hasattr(model, "sequence_len")
                and (sample.visibility[:, : model.sequence_len].sum() == 0)
            ):
                print(f"skipping batch {ind}")
                continue

            if "tapvid" in dataset_name:
                queries = sample.query_points.clone().float()

                queries = torch.stack(
                    [
                        queries[:, :, 0],  # t
                        queries[:, :, 2],  # x
                        queries[:, :, 1],  # y
                    ],
                    dim=2,
                ).to(device)  # B, N, 3 (t, x, y)
            else:
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory[:, 0, :, :1]),
                        sample.trajectory[:, 0],
                    ],
                    dim=2,
                ).to(device)  # B, N, 3 (t, x, y)

            if isinstance(model.model, CoTrackerThreeOnline):
                video = sample.video
                # NOTE: Comment out if don't want best-of-N / worst-of-N
                if num_samples > 1:
                    queries = queries.tile(num_samples, 1, 1)
                    video = video.tile(num_samples, 1, 1, 1, 1)

                online_model = CoTrackerOnlinePredictor(checkpoint=None)
                online_model.model = model.model
                online_model.step = model.model.window_len // 2
                online_model(
                    video_chunk=video,
                    is_first_step=True,
                    queries=queries,
                    add_support_grid=False,
                    iters=model.n_iters,
                )
                # Process the video
                for ind in range(
                    0, video.shape[1] - online_model.step, online_model.step
                ):
                    # TODO: Return confidence predictions as well
                    pred_tracks, pred_visibility, pred_confidence = online_model(
                        video_chunk=video[:, ind : ind + online_model.step * 2],
                        add_support_grid=False,
                        grid_size=0,
                        iters=model.n_iters,
                    )  # B T N 2,  B T N 1
                    # TODO: Pick best-of-N / worst-of-N here.
                    # NOTE: sample.video shape is (B, T, C, H, W), sample.trajectory shape is (B, T, N, 2), sample.visibility shape is (B, T, N), sample.valid shape is (B, T, N)
                    cur_frame = online_model.model.cur_frame
                    end_frame = online_model.model.end_frame
                    pred_tracks_, pred_visibility_, pred_confidence_ = best_of_n(
                        height=video.shape[-2],
                        width=video.shape[-1],
                        num_samples=num_samples,
                        use_oracle=use_oracle,
                        worst_of_n=worst_of_n,
                        trajs_pred=pred_tracks[:, cur_frame:end_frame],
                        vis_pred=pred_visibility[:, cur_frame:end_frame],
                        conf_pred=pred_confidence[:, cur_frame:end_frame],
                        trajs_gt=sample.trajectory[
                            :, cur_frame:end_frame
                        ],  # B, T_window, N, 2
                        vis_gt=sample.visibility[
                            :, cur_frame:end_frame
                        ],  # B, T_window, N
                        valids=sample.valid[:, cur_frame:end_frame],  # B, T_window, N
                    )
                    # TODO: Update online_model.model.online_coords_predicted, online_model.model.online_vis_predicted, online_model.model.online_conf_predicted
                    if ind == 0:
                        online_model.model.online_coords_predicted = pred_tracks_.tile(
                            num_samples, 1, 1, 1
                        )
                        online_model.model.online_vis_predicted = pred_visibility_.tile(
                            num_samples, 1, 1
                        )
                        online_model.model.online_conf_predicted = (
                            pred_confidence_.tile(num_samples, 1, 1)
                        )
                    else:
                        online_model.model.online_coords_predicted[
                            :, cur_frame:end_frame
                        ] = pred_tracks_.tile(num_samples, 1, 1, 1)
                        online_model.model.online_vis_predicted[
                            :, cur_frame:end_frame
                        ] = pred_visibility_.tile(num_samples, 1, 1)
                        online_model.model.online_conf_predicted[
                            :, cur_frame:end_frame
                        ] = pred_confidence_.tile(num_samples, 1, 1)
                pred_tracks = (pred_tracks[:1], pred_visibility[:1])
            else:
                pred_tracks = model(sample.video, queries)

            if "strided" in dataset_name:
                inv_video = sample.video.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

                pred_trj, pred_vsb = pred_tracks
                inv_pred_trj, inv_pred_vsb = model(inv_video, inv_queries)

                inv_pred_trj = inv_pred_trj.flip(1)
                inv_pred_vsb = inv_pred_vsb.flip(1)

                mask = pred_trj == 0

                pred_trj[mask] = inv_pred_trj[mask]
                pred_vsb[mask[:, :, :, 0]] = inv_pred_vsb[mask[:, :, :, 0]]

                pred_tracks = pred_trj, pred_vsb

            if dataset_name == "badja" or dataset_name == "fastcapture":
                seq_name = sample.seq_name[0]
            else:
                seq_name = str(ind)
            if ind % visualize_every == 0:
                vis.visualize(
                    sample.video,
                    pred_tracks[0] if isinstance(pred_tracks, tuple) else pred_tracks,
                    filename=dataset_name + "_" + seq_name,
                    writer=writer,
                    step=step,
                )
            self.compute_metrics(metrics, sample, pred_tracks, dataset_name)
        return metrics


def best_of_n(
    height,
    width,
    num_samples,
    use_oracle,
    worst_of_n,
    trajs_pred,
    vis_pred,
    conf_pred,
    trajs_gt,
    vis_gt,
    valids,
):
    num_samples, T_window, N, _ = trajs_pred.shape
    if num_samples > 1:
        # Stack the many different samples and pick the best one.
        if use_oracle:
            trajs_pred_ = trajs_pred[None]  # [1, num_samples, T_window, N, 2]
            traj_metrics = compute_traj_metrics(
                height=height,
                width=width,
                trajs_pred=trajs_pred_,
                trajs_gt=trajs_gt,
                vis_gt=vis_gt,
                valids=valids,
                keep_batch_dim=True,
                keep_N_dim=True,
            )  # [1, num_samples, N]
            if not worst_of_n:
                best_sample_ind = torch.max(
                    traj_metrics["d_all_avg"], dim=1
                ).indices.squeeze(0)  # [N]
            else:
                best_sample_ind = torch.min(
                    traj_metrics["d_all_avg"], dim=1
                ).indices.squeeze(0)  # [N]
        else:
            avg_conf_pred = (
                conf_pred[..., 0].sigmoid().mean(dim=1)[None]
            )  # [1, num_samples, N]
            if not worst_of_n:
                best_sample_ind = torch.max(avg_conf_pred, dim=1).indices.squeeze(
                    0
                )  # [N]
            else:
                best_sample_ind = torch.min(avg_conf_pred, dim=1).indices.squeeze(
                    0
                )  # [N]
        trajs_pred = torch.gather(
            trajs_pred,
            dim=0,
            index=repeat(best_sample_ind, "N -> 1 T N 2", T=T_window),
        )  # [1, T_window, N, 2]
        vis_pred = torch.gather(
            vis_pred,
            dim=0,
            index=repeat(best_sample_ind, "N -> 1 T N", T=T_window),
        )  # [1, T_window, N]
        conf_pred = torch.gather(
            conf_pred,
            dim=0,
            index=repeat(best_sample_ind, "N -> 1 T N", T=T_window),
        )  # [1, T_window, N]
    return trajs_pred, vis_pred, conf_pred


@torch.no_grad()
def compute_traj_metrics(
    height: int,
    width: int,
    trajs_pred: torch.Tensor,
    trajs_gt: torch.Tensor,
    vis_gt: torch.Tensor,
    valids: torch.Tensor,
    keep_batch_dim: bool = False,
    keep_N_dim: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute evaluation metrics from trajectory predictions, trajectory ground truths, track visibility ground
    truths, and valid masks.

    If `trajs_pred` is a `[B, I, T, N, 2]` tensor, the losses will be computed for each step of the refinement
    process and returned as a `[I]` tensor. Otherwise, if `trajs_pred` is a `[B, T, N, 2]` tensor, the losses will
    be computed for the final prediction and returned as a `[1]` tensor.

    :param video: A tensor of shape `[B, T, C, H, W]` containing the video.
    :param trajs_pred: A tensor of shape `[B, T, N, 2]` containing the predicted tracks. Can also provide a
    `[B, I, T, N, 2]` tensor, where `I` indexes through the iterative refinement process.
    :param trajs_gt: A tensor of shape `[B, T, N, 2]` containing the ground truth tracks.
    :param vis_gt: A tensor of shape `[B, T, N]` containing the visibility masks.
    :param valids: A tensor of shape `[B, T, N]` containing the valid masks.
    :param keep_batch_dim: Whether to keep the batch dimension in the output.
    :param keep_N_dim: Whether to keep the N dimension in the output.

    :raises ValueError: If the shapes of `trajs_pred` is invalid.

    :return: The computed losses as a dictionary, containing:
        - "L1_all": The L1 loss on all valid points. Also known as the Average Trajectory Error (ATE L1).
        - "L1_vis": The L1 loss on visible (and valid) points.
        - "L1_occ": The L1 loss on occluded (and valid) points.
        - "L2_all": The L2 loss on all valid points. Also known as the Average Trajectory Error (ATE L2).
        - "L2_vis": The L2 loss on visible (and valid) points.
        - "L2_occ": The L2 loss on occluded (and valid) points.
        - "d_all_1": The percentage of predicted (valid) points with their scaled ATE L2 < 1.
        - "d_all_2": The percentage of predicted (valid) points with their scaled ATE L2 < 2.
        - "d_all_4": The percentage of predicted (valid) points with their scaled ATE L2 < 4.
        - "d_all_8": The percentage of predicted (valid) points with their scaled ATE L2 < 8.
        - "d_all_16": The percentage of predicted (valid) points with their scaled ATE L2 < 16.
        - "d_all_avg": The percentage of predicted (valid) points with their scaled ATE L2 < `d_avg_threshold`,
            with `d_avg_threshold = [1, 2, 4, 8, 16]`, averaged over all thresholds.
        - "d_vis_1": The percentage of predicted (valid) visible points with their scaled ATE L2 < 1.
        - "d_vis_2": The percentage of predicted (valid) visible points with their scaled ATE L2 < 2.
        - "d_vis_4": The percentage of predicted (valid) visible points with their scaled ATE L2 < 4.
        - "d_vis_8": The percentage of predicted (valid) visible points with their scaled ATE L2 < 8.
        - "d_vis_16": The percentage of predicted (valid) visible points with their scaled ATE L2 < 16.
        - "d_vis_avg": The percentage of predicted (valid) visible points with their scaled ATE L2 <
            `d_avg_threshold`, with `d_avg_threshold = [1, 2, 4, 8, 16]`, averaged over all thresholds.
        - "d_occ_1": The percentage of predicted (valid) occluded points with their scaled ATE L2 < 1.
        - "d_occ_2": The percentage of predicted (valid) occluded points with their scaled ATE L2 < 2.
        - "d_occ_4": The percentage of predicted (valid) occluded points with their scaled ATE L2 < 4.
        - "d_occ_8": The percentage of predicted (valid) occluded points with their scaled ATE L2 < 8.
        - "d_occ_16": The percentage of predicted (valid) occluded points with their scaled ATE L2 < 16.
        - "d_occ_avg": The percentage of predicted (valid) occluded points with their scaled ATE L2 <
            `d_avg_threshold`, with `d_avg_threshold = [1, 2, 4, 8, 16]`, averaged over all thresholds.
        - "survival": The average survival rate of predicted (valid) points. Computed as the cumulative product of
            the percentage of predicted (valid) points with their scaled ATE L2 < 16.
            The average fraction of video frames until the tracker fails (detected when the tracking error exceeds 16 pixels).
        - "median_l2_all": The median trajectory error (L2), averaged over trajectories.
    """
    if trajs_pred.ndim == 5:
        B, I, T, N, D = trajs_pred.shape
        B1, T1, N1 = vis_gt.shape
        B2, T2, N2 = valids.shape
        assert trajs_gt.shape == (B, T, N, D)
        assert T == T1 and T == T2 and N == N1 and N == N2 and D == 2
        trajs_gt = repeat(trajs_gt, "B T N D -> B I T N D", I=I)  # [B, I, T, N, 2]
        vis_gt = repeat(vis_gt, "B T N -> B I T N", I=I)  # [B, I, T, N]
        valids = repeat(valids, "B T N -> B I T N", I=I)  # [B, I, T, N]
    elif trajs_pred.ndim == 4:
        I = 1  # noqa: E741
        B, T, N, D = trajs_gt.shape
        B1, T1, N1 = vis_gt.shape
        B2, T2, N2 = valids.shape
        assert trajs_pred.shape == trajs_gt.shape
        assert T == T1 and T == T2 and N == N1 and N == N2 and D == 2
        trajs_pred = trajs_pred.unsqueeze(1)  # [B, 1, T, N, 2]
        trajs_gt = trajs_gt.unsqueeze(1)  # [B, 1, T, N, 2]
        vis_gt = vis_gt.unsqueeze(1)  # [B, 1, T, N]
        valids = valids.unsqueeze(1)  # [B, 1, T, N]
    else:
        raise ValueError(f"Invalid shape for `trajs_pred`: {trajs_pred.shape}")

    H, W = height, width
    device = trajs_gt.device

    metrics = {}

    diff = trajs_pred - trajs_gt  # [B, I, T, N, 2]

    if not keep_N_dim:
        reduce_dim = (2, 3)
    else:
        reduce_dim = 2

    # Average Trajectory Error (L1)
    l1_dists = diff.abs().sum(dim=-1)  # [B, I, T, N]
    l1_all = reduce_masked_mean(l1_dists, valids, reduce_dim)  # [B, I]
    l1_vis = reduce_masked_mean(l1_dists, valids * vis_gt, reduce_dim)  # [B, I]
    l1_occ = reduce_masked_mean(l1_dists, valids * (1.0 - vis_gt), reduce_dim)  # [B, I]
    metrics["l1_all"] = l1_all
    metrics["l1_vis"] = l1_vis
    metrics["l1_occ"] = l1_occ

    # Average Trajectory Error (L2)
    l2_dists = LA.vector_norm(diff, ord=2, dim=-1)  # [B, I, T, N]
    l2_all = reduce_masked_mean(l2_dists, valids, reduce_dim)  # [B, I]
    l2_vis = reduce_masked_mean(l2_dists, valids * vis_gt, reduce_dim)  # [B, I]
    l2_occ = reduce_masked_mean(l2_dists, valids * (1.0 - vis_gt), reduce_dim)  # [B, I]
    metrics["l2_all"] = l2_all
    metrics["l2_vis"] = l2_vis
    metrics["l2_occ"] = l2_occ

    if keep_N_dim:
        d_sum_shape = [B, I, N]
    else:
        d_sum_shape = [B, I]

    d_all_sum = torch.zeros(*d_sum_shape, device=device)
    d_vis_sum = torch.zeros(*d_sum_shape, device=device)
    d_occ_sum = torch.zeros(*d_sum_shape, device=device)
    d_avg_thresholds = [1, 2, 4, 8, 16]
    scale_factor = torch.tensor(
        [[[[[W / 256.0, H / 256.0]]]]], device=device
    )  # [1, 1, 1, 1, 2]
    l2_dists_scaled = LA.vector_norm(diff / scale_factor, ord=2, dim=-1)  # [B, I, T, N]

    # Start tracking after the frame where the first visible point appears.
    # NOTE: This implies that the query frame is the first frame where the first visible point appears.
    _, first_visible_inds = torch.max(vis_gt, dim=2, keepdim=True)  # [B, I, 1, N]
    frame_inds = repeat(torch.arange(T, device=device), "t -> b i t n", b=B, i=I, n=N)
    start_tracking_mask = frame_inds > first_visible_inds  # [B, I, T, N]

    # Fraction of visible/occluded/all predicted points with scaled_ATE < d_avg_threshold pixels.
    # NOTE: start_tracking_mask will mask out the first frame where the first visible point appears.
    # TODO: start_tracking_mask should be based off of the query frame
    for d_avg_threshold in d_avg_thresholds:
        d_ = (l2_dists_scaled < d_avg_threshold).float()  # [B, I, T, N]

        d_all = (
            reduce_masked_mean(d_, valids * start_tracking_mask, reduce_dim) * 100.0
        )  # [B, I]
        d_all_sum += d_all
        metrics[f"d_all_{d_avg_threshold}"] = d_all

        d_vis = (
            reduce_masked_mean(d_, vis_gt * valids * start_tracking_mask, reduce_dim)
            * 100.0
        )  # [B, I]
        d_vis_sum += d_vis
        metrics[f"d_vis_{d_avg_threshold}"] = d_vis

        d_occ = (
            reduce_masked_mean(
                d_, (1 - vis_gt) * valids * start_tracking_mask, reduce_dim
            )
            * 100.0
        )  # [B, I]
        d_occ_sum += d_occ
        metrics[f"d_occ_{d_avg_threshold}"] = d_occ

    metrics["d_all_avg"] = d_all_sum / len(d_avg_thresholds)  # [B, I]
    metrics["d_vis_avg"] = d_vis_sum / len(d_avg_thresholds)  # [B, I]
    metrics["d_occ_avg"] = d_occ_sum / len(d_avg_thresholds)  # [B, I]

    # Survival rate.
    # TODO: start_tracking_mask should be used here too.
    survival_threshold = 16.0
    l2_dists_scaled_ok = (l2_dists_scaled <= survival_threshold).float()  # [B, I, T, N]
    l2_dists_scaled_ok[~valids.bool()] = (
        1.0  # Ensure that invalid points are always considered as surviving.
    )
    survival = l2_dists_scaled_ok.cumprod(dim=2)  # [B, I, T, N]
    metrics["survival"] = torch.mean(survival, dim=reduce_dim) * 100.0  # [B, I]

    # Median (scaled) Trajectory Error (L2)
    # TODO: start_tracking_mask should be used here too.
    median_l2_scaled, nan_mask = reduce_masked_median(
        l2_dists_scaled, valids, dim=2, keepdim=True
    )  # [B, I, 1, N]
    metrics["median_l2_all"] = median_l2_scaled.nanmean(dim=reduce_dim)  # [B, I]

    if not keep_batch_dim:
        for metric_name, metric_tensor in metrics.items():
            metrics[metric_name] = metric_tensor.mean(dim=0)

    return metrics
