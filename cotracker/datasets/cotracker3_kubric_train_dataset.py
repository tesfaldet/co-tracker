import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import albumentations as A
import numpy as np
import numpy.typing as npt
import PIL.Image
import torch
from albumentations import ReplayCompose
from PIL.Image import Resampling
from torch.utils.data import Dataset
from cotracker.datasets.utils import CoTrackerData


def farthest_point_sample_np(xyz: npt.NDArray, num_samples: int, deterministic: bool = False) -> npt.NDArray:
    """Farthest point sampling is a greedy algorithm for sampling a representative subset of points from a point cloud
    iteratively. Starting from a random point in the point cloud, it iteratively selects the point that is farthest
    away from the previously selected points.

    The selected points form a subset that approximates the original point cloud, with a lower number of points, but
    preserving the important features of the original point cloud. It is often used as a preprocessing step for point
    cloud processing tasks, such as point cloud classification, segmentation, and registration.

    The selected points also form the centroids of a Voronoi tessellation.

    :param xyz: NDArray of shape `[N, C]` where `N` is the number of points, and `C` is the point dimensionality. This
    is the input point cloud.
    :param num_samples: Number of points to sample from the point cloud.
    :param deterministic: If `True`, farthest point sampling is deterministic, i.e., the same set of points is selected
    every time the function is called with the same input point cloud. If `False`, the farthest point sampling is
    non-deterministic, i.e., the set of selected points may vary between calls with the same input point cloud.

    :return: NDArray of shape `[num_samples]` containing the indices of the sampled points in the input point cloud.
    """
    N, C = xyz.shape
    inds = np.zeros(num_samples, dtype=np.int32)
    distance = np.ones(N) * 1e10
    if deterministic:
        farthest = 0
    else:
        farthest = np.random.randint(0, N, dtype=np.int32)
    for i in range(num_samples):
        inds[i] = farthest
        centroid = xyz[farthest].reshape(1, C)
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)
        if num_samples > N:
            # if we need more samples, make them random
            distance += np.random.randn(*distance.shape)
    return inds


def guess_mode(data: torch.Tensor | npt.NDArray) -> Literal["L", "RGB", "RGBA"]:
    """Guesses image mode from data shape.

    :param data: Image data.

    :raises ValueError: If image mode cannot be guessed due to un-supported data shape.

    :return: Image mode; either ``"L"``, ``"RGB"``, or ``"RGBA"``.
    """
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray), "data must be a Tensor or ndarray"
    if data.shape[-1] == 1:
        return "L"
    if data.shape[-1] == 3:
        return "RGB"
    if data.shape[-1] == 4:
        return "RGBA"
    if data.ndim == 2:
        return "L"
    raise ValueError("Un-supported shape for image conversion %s" % list(data.shape))


class TAPVidKubricSubSeq(Dataset):
    """Data structure for the TAP-Vid-Kubric dataset.

    The TAP-Vid benchmark is composed of both real-world videos with accurate human annotations of point tracks, and
    synthetic videos with perfect ground-truth point tracks.

    This dataset consists of videos generated from the synthetic MOVi-E dataset introduced in Kubric. Each video
    consists of roughly 20 objects dropped into a synthetic scene, with physics from Bullet and raytraced rendering
    from Blender.

    The trajectories are normalized to [0, W/H] in the dataset, i.e., raster coords instead of pixel.
    https://github.com/google-deepmind/tapnet/blob/main/README.md#a-note-on-coordinates
    "Integer-corners" convention https://ppwwyyxx.com/blog/2021/Where-are-Pixels/.

    This dataset is for the CoTracker3 version of Kubric, found here:
        https://huggingface.co/datasets/facebook/CoTracker3_Kubric

    NOTE: The dataset must be processed by running the process_cotracker3_kubric.py script in the scripts folder.

    Dataset statistics:
    - 5,869 videos
    - 512 x 512 resolution.
    - 25 fps
    - 32,768 trajectories are generated per video.
    - 120 frames per video
    - Synthetic videos

    The dataset is stored in the following file and folder structure:
    ```
    CoTracker3_Kubric/
    ├── 0000/
    │   ├── 0000.npy
    │   ├── 0000_trajs_2d.npy
    │   ├── 0000_visibility.npy
    │   ├── 0000_with_rank.npz
    │   ├── depths
    │   │   ├── 000.npy
    │   │   .
    │   │   .
    │   │   .
    │   │   └── 119.npy
    │   └── frames
    │       ├── 000.png
    │       .
    │       .
    │       .
    │       └── 119.png
    │
    .
    .
    .
    │
    ├── 5867/
    │
    └── 5868/
    ```

    The processed dataset is stored in the following file and folder structure:
    ```
    CoTracker3_Kubric/
    ├── 0000/
    │   ├── frames
    │   │   ├── 000.png
    │   │   .
    │   │   .
    │   │   .
    │   │   └── 119.png
    │   ├── trajs_2d
    │   │   ├── trajs_2d_000.npy
    │   │   .
    │   │   .
    │   │   .
    │   │   └── trajs_2d_119.npy
    │   └── visibs
    │       ├── visib_000.npy
    │       .
    │       .
    │       .
    │       └── visib_119.npy
    │
    .
    .
    .
    │
    ├── 5867/
    │
    └── 5868/
    ```
    """

    URL = None
    MD5 = None
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".npy")

    DATASET_NAME = "CoTracker3_Kubric"
    ORIG_IMG_SIZE = (512, 512)  # H x W
    ORIG_SEQUENCE_LENGTH = 120  # Number of frames per video in the CoTracker3 version of Kubric. 24 for CoTracker v1.

    def __init__(
        self,
        root: str,
        crop_size: Tuple[int, int] = (384, 512),
        resize_dont_crop: bool = False,
        sequence_length: int = 64,
        query_first_frame_only: bool = True,
        num_trajectories: int = 768,
        frameskips: List[int] = [0, 1, 3],
        deterministic_fps: bool = False,
        use_augs: bool = False,
        geo_aug_prob: float = 0.7,
        reverse_prob: float = 0.5,
        pad_bounds: Tuple[int, int] = (0, 64),
        resize_limit: Tuple[float, float] = (0.25, 1.5),
        max_resize_delta: float = 0.1,
        max_crop_origin_delta: int = 20,
        h_flip_prob: float = 0.5,
        v_flip_prob: float = 0.5,
        photo_aug_prob: float = 0.7,
        eraser_prob: float = 0.2,
        eraser_bounds: Tuple[int, int] = (20, 300),
        eraser_max: int = 10,
        replacer_prob: float = 0.2,
        replacer_bounds: Tuple[int, int] = (20, 300),
        replacer_max: int = 10,
        colour_aug_prob: float = 0.2,
    ) -> None:
        """Initialize a `TAPVidKubricSubSeq` dataset.

        :param root: Root directory of the dataset.
        :param crop_size: The size (H x W) to crop the image sequences to. Default is `(384, 512)`.
        :param resize_dont_crop: Whether to resize the image sequences instead of cropping them. Default is `False`.
        :param sequence_length: The length of image sequences and trajectories sampled from the dataset.
            Default is `64` per sequence.
        :param query_first_frame_only: Whether to query only the first frame of the image sequence. Default is `True`.
        :param num_trajectories: The number of trajectories per image sequence sampled from the dataset.
            Default is `768`.
        :param frameskips: The frameskips to use when sampling image sequences from the dataset. A frameskip of `0`
            means no skipping between images in the sequence, while a frameskip of `1` means every other image is
            skipped, and so on. Default is `[0, 1, 3]`.
        :param deterministic_fps: Whether Farthest Point Sampling (FPS) of trajectories should be deterministic. FPS
            occurs when there are more trajectories remaining after filtering than the requested `num_trajectories`.
            Default is `False`.
        :param use_augs: Whether to use geometric and photometric augmentations. Default is `False`.
        :param geo_aug_prob: The probability of applying geometric augmentations to the image sequences. Default is
            `0.7`.
        :param reverse_prob: The probability of reversing the image sequences. Default is `0.5`.
        :param pad_bounds: The bounds for random border padding. Default is `(0, 64)`.
        :param resize_limit: The limits for random scaling. Default is `(0.25, 1.5)`.
        :param max_resize_delta: The maximum delta for random scaling. Default is `0.1`.
        :param max_crop_origin_delta: The maximum delta for random cropping. Default is `20`.
        :param h_flip_prob: The probability of randomly flipping the image sequences horizontally. Default is `0.5`.
        :param v_flip_prob: The probability of randomly flipping the image sequences vertically. Default is `0.5`.
        :param photo_aug_prob: The probability of applying photometric augmentations to the image sequences.
            Default is `0.7`.
        :param eraser_prob: The probability of applying the eraser augmentation to the image sequences.
            Default is `0.2`.
        :param eraser_bounds: The bounds for the eraser augmentation. Default is `(20, 300)`.
        :param eraser_max: The maximum number of times to apply the eraser augmentation. Default is `10`.
        :param replacer_prob: The probability of applying the replacer augmentation to the image sequences.
            Default is `0.2`.
        :param replacer_bounds: The bounds for the replacer augmentation. Default is `(20, 300)`.
        :param replacer_max: The maximum number of times to apply the replacer augmentation. Default is `10`.
        :param colour_aug_prob: The probability of applying colour augmentations to the image sequences.
            Default is `0.2`.

        :raises ValueError: If `sequence_length` is less than `2` or `num_trajectories` is less than `1`.
        """
        if sequence_length < 2:
            raise ValueError("sequence_length must be greater than 1.")
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be greater than 0.")

        self.root = Path(os.path.expanduser(root))

        self.crop_size = crop_size
        self.resize_dont_crop = resize_dont_crop
        self.sequence_length = sequence_length
        self.query_first_frame_only = query_first_frame_only
        self.num_trajectories = num_trajectories
        self.frameskips = frameskips
        self.deterministic_fps = deterministic_fps

        self.full_sequences, self.sampled_sequences = self._make_paths()

        self.use_augs = use_augs

        # Geometric augmentations.
        self.geo_aug_prob = geo_aug_prob
        self.reverse_prob = reverse_prob  # For random sequence reversing.
        self.pad_bounds = pad_bounds  # For random border padding.
        self.resize_limit = resize_limit  # For random scaling.
        self.max_resize_delta = max_resize_delta  # For random scaling.
        self.max_crop_origin_delta = max_crop_origin_delta  # For random cropping.
        self.h_flip_prob = h_flip_prob  # For random horizontal flipping.
        self.v_flip_prob = v_flip_prob  # For random vertical flipping.

        # Photometric augmentations.
        # TODO: consider switching to Kornia since they have native support for video transformations.
        self.photo_aug_prob = photo_aug_prob
        self.eraser_prob = eraser_prob
        self.eraser_bounds = eraser_bounds
        self.eraser_max = eraser_max
        self.replacer_prob = replacer_prob
        self.replacer_bounds = replacer_bounds
        self.replacer_max = replacer_max
        self.colour_aug_prob = colour_aug_prob
        self.colour_augmenter = A.ReplayCompose(
            [
                A.GaussNoise(p=0.2),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                    ],
                    p=0.2,
                ),
                A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
            ],
            p=0.8,
        )

    @property
    def samples(self) -> List[Tuple[List[str], List[str], List[str], List[int], str]]:
        """Get the samples in the dataset.

        :return: The samples in the dataset.
        """
        return self.sampled_sequences

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        :return: The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, ind: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        :param ind: Index of the sample.
        :return: A sample from the dataset and its corresponding target/label, if available.
        """
        rgb_paths, traj_2d_paths, visib_paths, path_inds, seq_name = self.samples[ind]

        # Load the images for the sampled image sequence. Convert to RGB if necessary. Type is uint8 [0, 255].
        rgbs = []
        for rgb_path in rgb_paths:
            with open(rgb_path, "rb") as f:
                img = PIL.Image.open(f).convert("RGB")
                rgbs.append(np.asarray(img))

        # Load the 2D trajectories for the sampled image sequence. Trajectories are (x,y), (0,0) is top-left.
        trajs = []
        for traj_2d_path in traj_2d_paths:
            trajs.append(np.load(traj_2d_path, allow_pickle=True).astype(np.float32))
        trajs = np.stack(trajs, axis=0)  # sequence_length, num_trajectories, 2

        # Load the visibility annotations for the sampled image sequence. Visibs contains False values for points that
        # are occluded. Visible and invisible points can be used for supervision.
        visibs = []
        for visib_path in visib_paths:
            visibs.append(np.load(visib_path, allow_pickle=True).astype(bool))
        visibs = np.stack(visibs, axis=0)  # sequence_length, num_trajectories
        visibs = ~visibs  # Invert visibility annotations, so that True means visible and False means invisible.

        valids = np.ones_like(visibs, dtype=visibs.dtype)  # [S, N]

        assert (
            len(rgbs) == self.sequence_length
            and trajs.shape[0] == self.sequence_length
            and visibs.shape[0] == self.sequence_length
        ), f"Expected sequence length of {self.sequence_length}, got the following: rgbs: {len(rgbs)}, trajs: {trajs.shape[0]}, visibs: {visibs.shape[0]}."

        # Replace infs and nans in trajectory points with 0.
        inf_nan_xy_inds = np.isinf(trajs) | np.isnan(trajs)
        trajs[inf_nan_xy_inds] = 0.0

        # Set points whose x or y is either inf or nan to be invisible and invalid.
        inf_nan_inds = inf_nan_xy_inds[..., 0] | inf_nan_xy_inds[..., 1]
        visibs[inf_nan_inds] = False
        valids[inf_nan_inds] = False

        # Photometric and geometric augmentations.
        if self.use_augs:
            # Apply photometric augmentations to the image sequence with a probability of `photo_aug_prob`.
            if np.random.rand() < self.photo_aug_prob:
                rgbs, trajs, visibs = self._add_photo_augs(rgbs, trajs, visibs)

            # Apply geometric augmentations to the image sequence with a probability of `geo_aug_prob`.
            if np.random.rand() < self.geo_aug_prob:
                rgbs, trajs, visibs = self._add_geo_augs(rgbs, trajs, visibs)
            else:
                if np.random.rand() < 0.5:
                    # Crop the image sequence to the specified crop size.
                    rgbs, trajs = self._crop(rgbs, trajs, self.crop_size)
                else:
                    # Resize the image sequence to the specified crop size.
                    rgbs, trajs = self._resize(rgbs, trajs, self.crop_size)
        else:
            if self.resize_dont_crop:
                # Resize the image sequence to the specified crop size.
                rgbs, trajs = self._resize(rgbs, trajs, self.crop_size)
            else:
                # Crop the image sequence to the specified crop size.
                rgbs, trajs = self._crop(rgbs, trajs, self.crop_size)

        H, W, C = rgbs[0].shape
        assert C == 3

        # Update visibility annotations, post augmentations. Do the following:
        #   1) Set points along the 1px edge of the image or out of bounds (OOB) to be invisible.
        for s in range(trajs.shape[0]):
            # Set points along the 1px edge of the image or out of bounds to be invisible.
            edge_oob_inds = np.logical_or(
                np.logical_or(trajs[s, :, 0] < 1, trajs[s, :, 0] > W - 1),
                np.logical_or(trajs[s, :, 1] < 1, trajs[s, :, 1] > H - 1),
            )
            visibs[s, edge_oob_inds] = False

            # Set points that are very out of bounds (past a 64px buffer zone) to be invalid.
            very_oob_inds = np.logical_or(
                np.logical_or(trajs[s, :, 0] < -64, trajs[s, :, 0] > W + 64),
                np.logical_or(trajs[s, :, 1] < -64, trajs[s, :, 1] > H + 64),
            )
            valids[s, very_oob_inds] = False

        if self.query_first_frame_only:
            # Remove trajectories that are not visible and valid in the first frame.
            vis_val_f0_inds = valids[0] & visibs[0]
            trajs = trajs[:, vis_val_f0_inds]
            visibs = visibs[:, vis_val_f0_inds]
            valids = valids[:, vis_val_f0_inds]
        else:
            # Remove trajectories that are not visible and valid in at least one frame.
            # vis_val_any_inds = np.any(valids & visibs, axis=0)
            # trajs = trajs[:, vis_val_any_inds]
            # visibs = visibs[:, vis_val_any_inds]
            # valids = valids[:, vis_val_any_inds]

            # Remove trajectories that are not visible and valid in at least sqrt(trajs.shape[0]) frames.
            mostly_vis_val_traj_inds = np.sum(visibs * valids, axis=0) >= np.sqrt(trajs.shape[0])
            trajs = trajs[:, mostly_vis_val_traj_inds]
            visibs = visibs[:, mostly_vis_val_traj_inds]
            valids = valids[:, mostly_vis_val_traj_inds]

        # After the filtering stage above, do the following:
        #   1) If there are more trajectories remaining than the requested `num_trajectories`, use farthest point
        #      sampling to sample a representative subset of `num_trajectories` trajectories.
        #   2) If the number of trajectories remaining is equal to the requested `num_trajectories`, use all
        #      trajectories.
        #   3) If there are fewer trajectories remaining than the requested `num_trajectories`, pad the number of
        #      trajectories to `num_trajectories` by concatenating invalid and invisible trajectories.
        if trajs.shape[1] > self.num_trajectories:
            # For each trajectory, compute its average position and velocity over the sequence.
            # Use these for farthest point sampling. This will result in a subset of trajectories that are
            # furthest away from each other in terms of average position and average velocity.
            xyv = np.concatenate([np.mean(trajs, axis=0), np.mean(trajs[1:] - trajs[:-1], axis=0)], axis=-1)

            farthest_traj_inds = farthest_point_sample_np(
                xyv, self.num_trajectories, deterministic=self.deterministic_fps
            )

            trajs_full = trajs[:, farthest_traj_inds]
            visibs_full = visibs[:, farthest_traj_inds].astype(np.float32)
            valids_full = valids[:, farthest_traj_inds].astype(np.float32)
        elif trajs.shape[1] == self.num_trajectories:
            trajs_full = trajs
            visibs_full = visibs.astype(np.float32)
            valids_full = valids.astype(np.float32)
        else:
            # Pad the number of trajectories to `num_trajectories` by adding invalid trajectories.
            N = trajs.shape[1]
            trajs_full_x = np.random.uniform(1, W - 1, size=(self.sequence_length, self.num_trajectories, 1))
            trajs_full_y = np.random.uniform(1, H - 1, size=(self.sequence_length, self.num_trajectories, 1))
            trajs_full = np.concatenate([trajs_full_x, trajs_full_y], axis=-1).astype(np.float32)
            visibs_full = np.zeros((self.sequence_length, self.num_trajectories), dtype=np.float32)
            valids_full = np.zeros((self.sequence_length, self.num_trajectories), dtype=np.float32)
            trajs_full[:, :N] = trajs
            visibs_full[:, :N] = visibs.astype(np.float32)
            valids_full[:, :N] = valids.astype(np.float32)

        rgbs = torch.from_numpy(np.stack(rgbs, 0).transpose(0, 3, 1, 2)).to(torch.uint8)  # [S, C, H, W]
        trajs = torch.from_numpy(trajs_full)  # [S, N, 2]
        visibs = torch.from_numpy(visibs_full)  # [S, N]
        valids = torch.from_numpy(valids_full)  # [S, N]

        # sample = {
        #     "seq_name": seq_name,
        #     "rgbs": rgbs,
        #     "trajs": trajs,
        #     "visibs": visibs,
        #     "valids": valids,
        # }

        gotit = True
        sample = (CoTrackerData(
            video=rgbs,
            trajectory=trajs,
            visibility=visibs,
            valid=valids,
            seq_name=seq_name,
        ), gotit)

        return sample

    def _add_geo_augs(
        self, rgbs: List[npt.NDArray], trajs: npt.NDArray, visibs: npt.NDArray
    ) -> Tuple[List[npt.NDArray], npt.NDArray, npt.NDArray]:
        """Apply geometric augmentations to the image sequence.

        Augmentations consist of random sequence reversing, random scaling, random cropping, and random horizontal and
        vertical flipping.

        :param rgbs: The image sequence to augment.
        :param trajs: The trajectory points for the image sequence.
        :param visibs: The visibility of the trajectory points for the image sequence.
        :return: The augmented image sequence, its associated augmented trajectory points, and associated visib labels.
        """
        sequence_length = len(rgbs)
        assert sequence_length == trajs.shape[0] == visibs.shape[0]

        in_dtype = rgbs[0].dtype

        # Make copies to avoid accidental in-place modifications.
        rgbs = [rgb.copy().astype(np.float32) for rgb in rgbs]
        trajs = trajs.copy()
        visibs = visibs.copy()

        # Start by resizing to a size larger than the crop size to allow for random cropping.
        rgbs, trajs = self._resize(rgbs, trajs, (self.crop_size[0] + 64, self.crop_size[1] + 64))

        ####################
        # Random reversing #
        ####################

        # Reverse the image sequence with a probability of `reverse_prob`.
        if np.random.rand() < self.reverse_prob:
            rgbs = rgbs[::-1]
            trajs = trajs[::-1]
            visibs = visibs[::-1]

        #########################
        # Random border padding #
        #########################

        # Padding will be consistent across frames.
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])  # Left border.
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])  # Right border.
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])  # Top border.
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])  # Bottom border.

        # Apply padding to each image.
        # Padding is applied in the format ((before, after), (before, after), (0, 0)) for (H, W, C).
        rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]

        # Since (0, 0) is located at the top-left corner, we add padding to the (x,y) trajectory points as such.
        trajs[..., 0] += pad_x0
        trajs[..., 1] += pad_y0

        ##################
        # Random scaling #
        ##################

        # Compute the scaling factor for the first image.
        scale = np.random.uniform(self.resize_limit[0], self.resize_limit[1])
        scale_x = scale_y = scale

        scale_delta_x = scale_delta_y = 0.0

        rgbs_scaled = []

        # Apply scaling to each image and its associated trajectory points.
        # Each frame's scaling is chosen based on the previous frame's scaling, with a small random delta.
        for s in range(sequence_length):
            if s == 1:
                # Scale the second frame based on the first frame's scaling with a small random delta.
                scale_delta_x = np.random.uniform(-self.max_resize_delta, self.max_resize_delta)
                scale_delta_y = np.random.uniform(-self.max_resize_delta, self.max_resize_delta)
            elif s > 1:
                # Smoothen the scaling deltas from frame 3 onward to avoid sudden jumps in scaling.
                scale_delta_x = (
                    scale_delta_x * 0.8 + np.random.uniform(-self.max_resize_delta, self.max_resize_delta) * 0.2
                )
                scale_delta_y = (
                    scale_delta_y * 0.8 + np.random.uniform(-self.max_resize_delta, self.max_resize_delta) * 0.2
                )

            # Compute the x and y scaling factors.
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # Bring the x and y scaling closer to each other to avoid wonky aspect ratios.
            scale_xy = (scale_x + scale_y) * 0.5
            scale_x = scale_x * 0.5 + scale_xy * 0.5
            scale_y = scale_y * 0.5 + scale_xy * 0.5

            # Clamp scaling to a reasonable range.
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)

            # Compute the new height and width of the image after scaling.
            H, W = rgbs[s].shape[:2]
            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # Clamp the new height and width to be at least slightly bigger than the crop size.
            # This is so that the random cropping (the next augmentation) can add diversity.
            H_new = np.clip(H_new, self.crop_size[0] + 16, None)
            W_new = np.clip(W_new, self.crop_size[1] + 16, None)

            # Recompute the scale factors based on the clamped new height and width above.
            scale_x = W_new / float(W)
            scale_y = H_new / float(H)

            # Apply scaling to the image and trajectory points for that image.
            rgb = PIL.Image.fromarray(rgbs[s].astype(np.uint8), mode=guess_mode(rgbs[s]))
            rgb_scaled = rgb.resize(size=(W_new, H_new), resample=Resampling.LANCZOS)
            rgbs_scaled.append(np.array(rgb_scaled).astype(np.float32))
            trajs[s, :, 0] *= scale_x
            trajs[s, :, 1] *= scale_y

        rgbs = rgbs_scaled

        ###################
        # Random cropping #
        ###################

        # Retrieve trajectories whose points are visible in the first frame.
        vis_f1_trajs = trajs[:, visibs[0]]

        if vis_f1_trajs.shape[1] > 0:
            # If there are visible trajectory points in the first frame, compute the average x and y position of the
            # trajectories' visible first points. This will be used to compute the new origin for cropping the
            # the first frame. This will result in the center of the cropped first frame being the average visible
            # trajectory point in the first frame.
            crop_mid_x = np.mean(vis_f1_trajs[0, :, 0])
            crop_mid_y = np.mean(vis_f1_trajs[0, :, 1])
        else:
            # If there are no visible trajectory points in the first frame, the crop extents will be used to compute
            # the new origin point for cropping the first frame. This will result in the origin of the cropped
            # first frame being the center of the first frame.
            crop_mid_y = self.crop_size[0]
            crop_mid_x = self.crop_size[1]

        # Compute the origin point for the cropped first image.
        crop_origin_x = int(crop_mid_x - self.crop_size[1] // 2)
        crop_origin_y = int(crop_mid_y - self.crop_size[0] // 2)

        crop_origin_delta_x = crop_origin_delta_y = 0

        # Apply cropping to each frame and its associated trajectory points.
        # For each frame, a new image origin is chosen and the image is cropped to the crop size from that origin.
        # Each new origin is chosen based on the previous frame's origin, with a small random offset.
        for s in range(sequence_length):
            if s == 1:
                # Compute the crop origin of the second frame based on the first frame's crop origin with a small
                # random delta.
                crop_origin_delta_x = np.random.randint(-self.max_crop_origin_delta, self.max_crop_origin_delta)
                crop_origin_delta_y = np.random.randint(-self.max_crop_origin_delta, self.max_crop_origin_delta)
            elif s > 1:
                # Smoothen the crop origin deltas from frame 3 onward to avoid sudden jumps in crop locations.
                crop_origin_delta_x = int(
                    crop_origin_delta_x * 0.8
                    + np.random.randint(-self.max_crop_origin_delta, self.max_crop_origin_delta + 1) * 0.2
                )
                crop_origin_delta_y = int(
                    crop_origin_delta_y * 0.8
                    + np.random.randint(-self.max_crop_origin_delta, self.max_crop_origin_delta + 1) * 0.2
                )

            # Compute the origin point for the cropped image.
            crop_origin_x = crop_origin_x + crop_origin_delta_x
            crop_origin_y = crop_origin_y + crop_origin_delta_y

            H, W = rgbs[s].shape[:2]

            if H == self.crop_size[0]:
                # If the height of the image is the same as the crop height, don't crop vertically.
                crop_origin_y = 0
            else:
                # Clamp the y coord of the cropped image's origin so that the vertical crop is within the vertical
                # bounds of the image.
                crop_origin_y = np.clip(crop_origin_y, 0, H - self.crop_size[0] - 1)

            if W == self.crop_size[1]:
                # If the width of the image is the same as the crop width, don't crop horizontally.
                crop_origin_x = 0
            else:
                # Clamp the x coord of the cropped image's origin so that the horizontal crop is within the horizontal
                # bounds of the image.
                crop_origin_x = np.clip(crop_origin_x, 0, W - self.crop_size[1] - 1)

            # Apply cropping to the image and trajectory points for that image.
            rgbs[s] = rgbs[s][
                crop_origin_y : crop_origin_y + self.crop_size[0], crop_origin_x : crop_origin_x + self.crop_size[1]
            ]
            trajs[s, :, 0] -= crop_origin_x
            trajs[s, :, 1] -= crop_origin_y

        ###################
        # Random flipping #
        ###################

        h_flipped = False
        v_flipped = False

        # Randomly flip the image sequence horizontally and/or vertically.
        if np.random.rand() < self.h_flip_prob:
            h_flipped = True
            rgbs = [rgb[:, ::-1] for rgb in rgbs]
        if np.random.rand() < self.v_flip_prob:
            v_flipped = True
            rgbs = [rgb[::-1] for rgb in rgbs]

        # Flip the trajectories based on how the image sequence was flipped.
        H, W = rgbs[0].shape[:2]
        if h_flipped:
            trajs[..., 0] = W - trajs[..., 0]
        if v_flipped:
            trajs[..., 1] = H - trajs[..., 1]

        rgbs = [rgb.astype(in_dtype) for rgb in rgbs]

        return rgbs, trajs, visibs

    def _add_photo_augs(
        self, rgbs: List[npt.NDArray], trajs: npt.NDArray, visibs: npt.NDArray
    ) -> Tuple[List[npt.NDArray], npt.NDArray, npt.NDArray]:
        """Apply photometric augmentations to the image sequence.

        :param rgbs: The image sequence to augment.
        :param trajs: The trajectory points for the image sequence.
        :param visibs: The visibility of the trajectory points for the image sequence.
        :return: The augmented image sequence, its associated augmented trajectory points, and associated visib labels.
        """
        sequence_length = len(rgbs)
        assert sequence_length == trajs.shape[0] == visibs.shape[0]

        in_dtype = rgbs[0].dtype

        # Make copies to avoid accidental in-place modifications.
        rgbs = [rgb.copy().astype(np.float32) for rgb in rgbs]
        trajs = trajs.copy()
        visibs = visibs.copy()

        #############################
        # Random occlusion (eraser) #
        #############################

        H, W = rgbs[0].shape[:2]
        for i in range(1, sequence_length):
            if np.random.rand() < self.eraser_prob:
                for _ in range(np.random.randint(1, self.eraser_max + 1)):  # number of times to occlude
                    xc = np.random.randint(0, W)
                    yc = np.random.randint(0, H)
                    dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                    x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                    y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                    y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                    mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                    rgbs[i][y0:y1, x0:x1, :] = mean_color

                    occ_inds = np.logical_and(
                        np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                        np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                    )
                    visibs[i, occ_inds] = 0

        ###############################
        # Random occlusion (replacer) #
        ###############################

        rgbs_alt = [rgb.astype(np.uint8) for rgb in rgbs]
        rgbs_alt = self._augment_video(self.colour_augmenter, video=rgbs_alt)
        rgbs_alt = self._augment_video(self.colour_augmenter, video=rgbs_alt)
        rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]

        for i in range(1, sequence_length):
            if np.random.rand() < self.replacer_prob:
                for _ in range(np.random.randint(1, self.replacer_max + 1)):  # number of times to occlude
                    xc = np.random.randint(0, W)
                    yc = np.random.randint(0, H)
                    dx = np.random.randint(self.replacer_bounds[0], self.replacer_bounds[1])
                    dy = np.random.randint(self.replacer_bounds[0], self.replacer_bounds[1])
                    x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                    x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                    y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                    y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                    w = x1 - x0
                    h = y1 - y0
                    y00 = np.random.randint(0, H - h)
                    x00 = np.random.randint(0, W - w)
                    fr = np.random.randint(0, sequence_length)
                    rep = rgbs_alt[fr][y00 : y00 + h, x00 : x00 + w, :]
                    rgbs[i][y0:y1, x0:x1, :] = rep

                    occ_inds = np.logical_and(
                        np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                        np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                    )
                    visibs[i, occ_inds] = 0

        #####################
        # Random colour aug #
        #####################

        if np.random.rand() < self.colour_aug_prob:
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]
            rgbs = self._augment_video(self.colour_augmenter, video=rgbs)

        rgbs = [rgb.astype(in_dtype) for rgb in rgbs]

        return rgbs, trajs, visibs

    # TODO: consider switching to Kornia since they have native support for video transformations.
    @staticmethod
    def _augment_video(transform: ReplayCompose, video: List[npt.NDArray], replay: bool = False) -> List[npt.NDArray]:
        """Apply a set of augmentations to an image sequence. The augmentations can be replayed across the image
        sequence or be random per image.

        :param transform: The set of augmentations to apply to the image sequence. Must be an instance of
        ReplayCompose.
        :param video: The image sequence to augment.
        :param replay: Whether to replay the augmentations across the image sequence. Default is `False`.
        :return: The augmented image sequence.
        """
        assert isinstance(transform, ReplayCompose), "transform must be an instance of ReplayCompose."
        data = [transform(image=video[0])]
        replay_transform = partial(ReplayCompose.replay, data[0]["replay"]) if replay else transform
        for i in range(1, len(video)):
            data.append(replay_transform(image=video[i]))
        data = [d["image"] for d in data]
        return data

    def _crop(
        self, rgbs: List[npt.NDArray], trajs: npt.NDArray, size: Tuple[int, int], random: bool = False
    ) -> Tuple[List[npt.NDArray], npt.NDArray]:
        """Crop an image sequence to a specified crop size.

        :param rgbs: The image sequence to crop.
        :param trajs: The trajectory points for the image sequence.
        :param size: The size (H x W) to crop the image sequences to.
        :param random: Whether to randomly pick the crop origin point. Default is `False`.
        :return: The cropped image sequence and its associated cropped trajectory points.
        """
        assert len(rgbs) == trajs.shape[0]

        H, W = rgbs[0].shape[:2]
        crop_H, crop_W = size

        if random:
            crop_origin_y = np.random.randint(0, H - crop_H)
            crop_origin_x = np.random.randint(0, W - crop_W)
        else:
            crop_origin_y = (H - crop_H) // 2
            crop_origin_x = (W - crop_W) // 2

        rgbs_out = [rgb[crop_origin_y : crop_origin_y + crop_H, crop_origin_x : crop_origin_x + crop_W] for rgb in rgbs]

        trajs_out = trajs.copy()
        trajs_out[:, :, 0] -= crop_origin_x
        trajs_out[:, :, 1] -= crop_origin_y

        return rgbs_out, trajs_out

    def _resize(
        self, rgbs: List[npt.NDArray], trajs: npt.NDArray, size: Tuple[int, int]
    ) -> Tuple[List[npt.NDArray], npt.NDArray]:
        """Resize an image sequence to a specified crop size.

        :param rgbs: The image sequence to resize.
        :param trajs: The trajectory points for the image sequence.
        :param size: The size (H x W) to resize the image sequences to.
        :return: The resized image sequence and its associated resized trajectory points.
        """
        assert len(rgbs) == trajs.shape[0]

        in_dtype = rgbs[0].dtype
        H, W = rgbs[0].shape[:2]
        H_new, W_new = size

        scale_x = W_new / float(W)
        scale_y = H_new / float(H)

        # Convert from np.float32 [0, 255] to PIL.Image to allow for better resizing quality.
        rgbs_out = [PIL.Image.fromarray(rgb.astype(np.uint8), mode=guess_mode(rgb)) for rgb in rgbs]

        # Resize.
        rgbs_out = [rgb.resize(size=(W_new, H_new), resample=Resampling.LANCZOS) for rgb in rgbs_out]

        # Convert back to original dtype.
        rgbs_out = [np.array(rgb).astype(in_dtype) for rgb in rgbs_out]

        trajs_out = trajs.copy()
        trajs_out[..., 0] *= scale_x
        trajs_out[..., 1] *= scale_y

        return rgbs_out, trajs_out

    def _make_paths(
        self,
    ) -> Tuple[
        Dict[str, Dict[str, List[str]]],
        List[Tuple[List[str], List[str], List[str], List[str], List[int], str]],
    ]:
        """Create dicts containing relevant filepaths for the train, val, and test sets.

        `full_sequences` is a dict containing the full image sequence paths, traj_2d paths, and visibs paths.

        ```python
        full_sequences = {
            "000": {
                "rgb_paths": [
                    "path/to/0000/frames/000.png",
                    "path/to/0000/frames/000.png",
                    ...
                ],
                "traj_2d_paths": [
                    "path/to/0000/trajs_2d/traj_2d_00000.npy",
                    "path/to/0000/trajs_2d/traj_2d_00001.npy",
                    ...
                ],
                "visib_paths": [
                    "path/to/0000/visibs/visib_00000.npy",
                    "path/to/0000/visibs/visib_00001.npy",
                    ...
                ],
            },
            "001": { ... },
            ...
        }
        ```

        `sampled_sequences` is a list containing image sequences sampled from `full_sequences` according to
        `self.sequence_length` and `self.frameskips`. Each sampled sequence is a tuple containing the image paths for
        the sequence, the trajs_2d paths, the visibs paths, indices describing each image path's indexed location
        within the full image sequence, and the sequence name.

        ```python
        sampled_sequences = [
                (
                    [
                        "path/to/0000/frames/000.png",
                        "path/to/0000/frames/002.png",
                        ...
                    ],
                    [
                        "path/to/0000/trajs_2d/traj_2d_00000.npy",
                        "path/to/0000/trajs_2d/traj_2d_00001.npy",
                        ...
                    ],
                    [
                        "path/to/0000/visibs/visib_00000.npy",
                        "path/to/0000/visibs/visib_00001.npy",
                        ...
                    ],
                    [0, 1, ...],
                    "0000",
                ),
        ]
        ```

        :return: A tuple containing the full sequences and sampled sequences.
        """
        sequences = sorted([seq for seq in self.root.iterdir() if seq.is_dir() and seq.name.isdigit()])

        # Build full sequences for the train, val, and test sets.
        full_sequences = {
            seq.name: {
                "rgb_paths": sorted(
                    [str(rgb_path) for rgb_path in (seq / "frames").iterdir() if self._is_valid_file(str(rgb_path))]
                ),
                "traj_2d_paths": sorted(
                    [
                        str(traj_2d_path)
                        for traj_2d_path in (seq / "trajs_2d").iterdir()
                        if self._is_valid_file(str(traj_2d_path))
                    ]
                ),
                "visib_paths": sorted(
                    [
                        str(visib_path)
                        for visib_path in (seq / "visibs").iterdir()
                        if self._is_valid_file(str(visib_path))
                    ]
                ),
            }
            for seq in sequences
        }

        sampled_sequences = []  # ([rgb_paths], [traj_2d_paths], [visib_paths], [inds], seq_name)

        # Build sampled sequences for the train set.
        for seq_name, seq_paths in full_sequences.items():
            full_seq_len = len(seq_paths["rgb_paths"])
            if self.sequence_length > full_seq_len:
                continue
            for skip in self.frameskips:
                for start_ind in range(full_seq_len - (self.sequence_length - 1) * (skip + 1)):
                    end_ind = start_ind + (self.sequence_length - 1) * (skip + 1)
                    rgb_path_inds = list(range(start_ind, end_ind + 1, skip + 1))
                    sampled_sequences.append(
                        (
                            [seq_paths["rgb_paths"][ind] for ind in rgb_path_inds],
                            [seq_paths["traj_2d_paths"][ind] for ind in rgb_path_inds],
                            [seq_paths["visib_paths"][ind] for ind in rgb_path_inds],
                            rgb_path_inds,
                            seq_name,
                        )
                    )

        return full_sequences, sampled_sequences

    def _is_valid_file(self, filename: str) -> bool:
        """Checks if a file is an allowed extension.

        :param filename: Path to a file.

        :return: `True` if the filename ends with one of the extensions listed in `IMG_EXTENSIONS`.
        """
        return filename.lower().endswith(
            self.IMG_EXTENSIONS if isinstance(self.IMG_EXTENSIONS, str) else tuple(self.IMG_EXTENSIONS)
        )
