# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import time
import argparse

from typing import Optional
from cotracker.datasets.utils import dataclass_to_cuda_
from tqdm import tqdm

from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline

from cotracker.datasets.utils import collate_fn
from cotracker.models.evaluation_predictor import EvaluationPredictor

from cotracker.models.build_cotracker import build_cotracker


@torch.no_grad()
def evaluate_compute(
    model,
    test_dataloader: torch.utils.data.DataLoader,
    N,
):

    avg_duration = 0.0
    avg_duration_per_frame = 0.0
    num_batches = len(test_dataloader)

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
            hasattr(model, "sequence_len")
            and (sample.visibility[:, : model.sequence_len].sum() == 0)
        ):
            print(f"skipping batch {ind}")
            continue

        B = sample.video.shape[0]

        queried_coords = torch.randn(B, N, 2, device=device)  # [B, N, 2]
        queried_frames = torch.zeros(B, N, 1, device=device)
        queries = torch.cat([queried_coords, queried_frames], dim=2)  # [B, N, 3]

        start = time.time()
        num_processed_frames = 0

        if isinstance(model.model, CoTrackerThreeOnline):
            online_model = CoTrackerOnlinePredictor(checkpoint=None)
            online_model.model = model.model
            online_model.step = model.model.window_len // 2
            online_model(
                video_chunk=sample.video,
                is_first_step=True,
                queries=queries,
                add_support_grid=False,
                iters=model.n_iters,
            )
            # Process the video
            for ind in range(
                0, sample.video.shape[1] - online_model.step, online_model.step
            ):
                pred_tracks, pred_visibility = online_model(
                    video_chunk=sample.video[:, ind : ind + online_model.step * 2],
                    add_support_grid=False,
                    grid_size=0,
                    iters=model.n_iters,
                )  # B T N 2,  B T N 1
                num_processed_frames += online_model.step
        else:
            pred_tracks = model(sample.video, queries)
            num_processed_frames += sample.video.shape[1]

        end = time.time()
        duration = end - start
        duration_per_frame = duration / num_processed_frames

        print(f"Processed sequence ({ind}) in {duration:.4f} seconds ({duration_per_frame:.4f}s per frame).")

        avg_duration += duration
        avg_duration_per_frame += duration_per_frame

    avg_duration /= num_batches
    avg_duration_per_frame /= num_batches

    print(f"Average duration per minibatch: {avg_duration:.4f} seconds.")
    print(f"Average duration per frame: {avg_duration_per_frame:.4f} seconds.")

    return duration, duration_per_frame


def run_eval(args):
    cotracker_model = build_cotracker(
        "/home/mila/m/mattie.tesfaldet/Projects/co-tracker/checkpoints/model_cotracker_three_400000_pod.pth", offline=args.offline, window_len=16, v2=False
    )

    # Creating the EvaluationPredictor object
    predictor = EvaluationPredictor(
        cotracker_model,
        grid_size=0,
        local_grid_size=0,
        sift_size=0,
        single_point=False,
        num_uniformly_sampled_pts=0,
        n_iters=6,
        local_extent=0,
        interp_shape=(384, 512),
    )

    if torch.cuda.is_available():
        predictor.model = predictor.model.cuda()

    # Setting the random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Constructing the specified dataset
    curr_collate_fn = collate_fn
    from cotracker.datasets.tap_vid_datasets import TapVidDataset

    test_dataset = TapVidDataset(
        dataset_type="davis",
        data_root="/network/datasets/tapvid.var/tapvid_extract/data/tapvid_davis/tapvid_davis.pkl",
        queried_first=True,
        resize_to=[384, 512],
    )

    # Creating the DataLoader object
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=5,
        collate_fn=curr_collate_fn,
    )

    duration, duration_per_frame = evaluate_compute(
        predictor, test_dataloader, N=args.num_queries
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", default=False, action="store_true")
    parser.add_argument(
        "--num_queries", type=int, default=1,
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    run_eval(args)
