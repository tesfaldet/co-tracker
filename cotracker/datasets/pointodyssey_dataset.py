import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import numpy.typing as npt
import torch
import PIL.Image
from PIL.Image import Resampling
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, verify_str_arg
from cotracker.datasets.utils import CoTrackerData


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


class PointOdyssey(Dataset):
    """Data structure for the Point Odyssey dataset.

    This differs from the PointOdyssey dataset class in that it returns the full sequence of images and trajectories,
    has no photometric and geometric augmentations (aside from a final resizing step), and has a simpler trajectory
    filtering stage. This dataset is intended to be used for evaluation purposes.

    PointOdyssey is a large-scale synthetic dataset for the training and evaluation of long-term fine-grained tracking
    algorithms. The dataset contains diversity via randomizing character appearance, motion profiles, materials,
    lighting, 3D assets, and atmospheric effects.

    The trajectories are normalized to [0, W/H] in the dataset, i.e., raster coords instead of pixel.
    "Integer-centers" convention https://ppwwyyxx.com/blog/2021/Where-are-Pixels/.

    Dataset statistics:
    - 102 videos
    - 540 × 960 resolution
    - 30 fps
    - 18,700 trajectories per video on average
    - 2,035 frames per video on average
    - 166K training frames (78 sequences)
    - 24K validation frames (12 sequences)
    - 26K test frames (12 sequences)
    - 4.9×10^10 total point annotations.

    Additional attributes include:
    - Depth & normals
    - Segmentation masks
    - Retargeted motion
    - Scene randomization
    - Multiple views
    - Continuous
    - Object-object interaction
    - Human-object interaction
    - Human-human interaction

    ```
    point_odyssey/
    ├── train/
    │   ├── ani/
    │   │   ├── scene_info.json
    │   │   ├── depths/
    │   │   │   ├── depth_00000.png
    │   │   │   ├── d..
    │   │   │   └── ...
    │   │   ├── extrinsics/
    │   │   │   ├── extrinsic_00000.npy
    │   │   │   ├── e..
    │   │   │   └── ...
    │   │   ├── intrinsics/
    │   │   │   ├── intrinsic_00000.npy
    │   │   │   ├── i..
    │   │   │   └── ...
    │   │   ├── masks/
    │   │   │   ├── mask_00000.png
    │   │   │   ├── m..
    │   │   │   └── ...
    │   │   ├── normals/
    │   │   │   ├── normal_00000.jpg
    │   │   │   ├── n..
    │   │   │   └── ...
    │   │   ├── rgbs/
    │   │   │   ├── rgb_00000.jpg
    │   │   │   ├── r..
    │   │   │   └── ...
    │   │   ├── trajs_2d/
    │   │   │   ├── traj_2d_00000.npy
    │   │   │   ├── t..
    │   │   │   └── ...
    │   │   ├── trajs_3d/
    │   │   │   ├── traj_3d_00000.npy
    │   │   │   ├── t..
    │   │   │   └── ...
    │   │   ├── valids/
    │   │   │   ├── valid_00000.npy
    │   │   │   ├── v..
    │   │   │   └── ...
    │   │   └── visibs/
    │   │       ├── visib_00000.npy
    │   │       ├── v..
    │   │       └── ...
    │   ├── ani.mp4
    │   ├── ani2/
    │   │   ├── scene_info.json
    │   │   ├── depths/
    │   │   ├── masks/
    │   │   ├── normals/
    │   │   ├── rgbs/
    │   │   ├── trajs_2d/
    │   │   ├── trajs_3d/
    │   │   ├── valids/
    │   │   └── visibs/
    │   ├── ani2.mp4
    │
    ├── test/
    │
    ├── val/
    │
    └── sample/
    ```
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".npy")
    TRAIN_URL = None
    VAL_URL = None
    TEST_URL = None
    DATASET_NAME = "point_odyssey_v1.4"
    TRAIN_MD5 = None
    VAL_MD5 = None
    TEST_MD5 = None

    ORIG_IMG_SIZE = (540, 960)

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_trajectories: int = 256,
        first_n_frames: int = -1,
        load_rgbs: bool = False,
        resize_size: Tuple[int, int] = (384, 512),
    ) -> None:
        """Initialize a `PointOdysseyFullSeq` dataset.

        :param root: Root directory of the dataset.
        :param split: One of `"train"`, `"test"`, or `"val"`. Default is `"train"`.
        :param num_trajectories: The number of trajectories per image sequence sampled from the dataset.
        Default is `256`.
        :param first_n_frames: The number of frames to use from the start of each sequence. Default is `100`.
        :param load_rgbs: Whether to load the RGB images. Default is `False`.
        :param resize_size: The size (H x W) to resize the image sequences to. Defaults to `(384, 512)`.

        :raises ValueError: If `num_trajectories` is less than 1.
        """
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be greater than 0.")

        self.root = Path(os.path.expanduser(root))
        self.split = verify_str_arg(split, "split", ("train", "test", "val"))

        self.num_trajectories = num_trajectories
        self.first_n_frames = first_n_frames
        self.load_rgbs = load_rgbs
        self.resize_size = resize_size

        self.train_path = self.root / "train"
        self.val_path = self.root / "val"
        self.test_path = self.root / "test"

        self.sequences = self._make_paths()

    @property
    def split_folder(self) -> str:
        """Get the path to the folder for the current split.

        :return: A string containing the path to the current split folder.
        """
        return str(self.root / self.split)

    @property
    def samples(self) -> List[Tuple[List[str], List[str], List[str], List[str], str]]:
        """Get the samples in the dataset.

        :return: The samples in the dataset.
        """
        return self.sequences[self.split]

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
        rgb_paths, traj_2d_paths, valid_paths, visib_paths, seq_name = self.samples[ind]

        rgbs = []
        if self.load_rgbs:
            # Load the images for the sampled image sequence. Convert to RGB if necessary. Type is uint8 [0, 255].
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

        # Load the validity annotations for the sampled image sequence. Valids contains False values for points that
        # are far out of bounds or contain NaNs or inf values. Invalid points are not used for supervision.
        # Basically, a point is considered invalid if a human would not be able to locate it.
        valids = []
        for valid_path in valid_paths:
            valids.append(np.load(valid_path, allow_pickle=True).astype(bool))
        valids = np.stack(valids, axis=0)  # sequence_length, num_trajectories

        assert (
            trajs.shape[0] == visibs.shape[0] == valids.shape[0]
        ), f"Sequence length mismatch: trajs: {trajs.shape[0]}, visibs: {visibs.shape[0]}, valids: {valids.shape[0]}."

        # Replace infs and nans in trajectory points with 0.
        inf_nan_xy_inds = np.isinf(trajs) | np.isnan(trajs)
        trajs[inf_nan_xy_inds] = 0.0

        # Set points whose x or y is either inf or nan to be invisible and invalid.
        inf_nan_inds = inf_nan_xy_inds[..., 0] | inf_nan_xy_inds[..., 1]
        visibs[inf_nan_inds] = False
        valids[inf_nan_inds] = False

        if len(rgbs) > 0:
            # Resize the image sequence to the specified size.
            rgbs, trajs = self._resize(rgbs, trajs, self.resize_size)
            H, W, C = rgbs[0].shape
            assert C == 3
        else:
            H, W = self.ORIG_IMG_SIZE

        # Update visibility and validity annotations.
        for s in range(trajs.shape[0]):
            # Set points along the 1px edge of the image or out of bounds to be invisible and invalid.
            edge_oob_inds = np.logical_or(
                np.logical_or(trajs[s, :, 0] < 1, trajs[s, :, 0] > W - 1),
                np.logical_or(trajs[s, :, 1] < 1, trajs[s, :, 1] > H - 1),
            )
            visibs[s, edge_oob_inds] = False
            valids[s, edge_oob_inds] = False

        # Remove trajectories that are not visible and valid in the first frame.
        vis_val_f0_inds = valids[0] & visibs[0]
        trajs = trajs[:, vis_val_f0_inds]
        visibs = visibs[:, vis_val_f0_inds]
        valids = valids[:, vis_val_f0_inds]

        N = trajs.shape[1]
        assert N > 0, "No valid trajectories remaining after filtering."

        # If there are more trajectories remaining than requested, pick `num_trajectories` trajectories along evenly
        # spaced indices from the remaining trajectories.
        if N > self.num_trajectories:
            inds = np.linspace(start=0, stop=N - 1, num=self.num_trajectories).astype(np.int32)
            trajs = trajs[:, inds]
            visibs = visibs[:, inds]
            valids = valids[:, inds]

        # Convert from bool to float.
        visibs = visibs.astype(np.float32)
        valids = valids.astype(np.float32)

        if len(rgbs) > 0:
            rgbs = torch.from_numpy(np.stack(rgbs, 0).transpose(0, 3, 1, 2)).to(torch.uint8)  # [S, C, H, W]
        trajs = torch.from_numpy(trajs)  # [S, N, 2]
        visibs = torch.from_numpy(visibs)  # [S, N]
        valids = torch.from_numpy(valids)  # [S ,N]

        gotit = True if N > 0 else False
        sample = (CoTrackerData(
            video=rgbs,
            trajectory=trajs,
            visibility=visibs,
            valid=valids,
            seq_name=seq_name,
        ), gotit)

        # sample = {
        #     "seq_name": seq_name,
        #     "rgb_paths": rgb_paths,
        #     "rgbs": rgbs,
        #     "trajs": trajs,
        #     "visibs": visibs,
        #     "valids": valids,
        # }

        return sample

    def _resize(
        self, rgbs: List[npt.NDArray], trajs: npt.NDArray, size: Tuple[int, int]
    ) -> Tuple[List[npt.NDArray], npt.NDArray]:
        """Resize an image sequence to a specified crop size.

        :param rgbs: The image sequence to resize.
        :param trajs: The trajectory points for the image sequence.
        :param size: The size (H x W) to resize the image sequences to.
        :return: The resized image sequence and its associated resized trajectory points.
        """
        T = trajs.shape[0]

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        H_new, W_new = size

        scale_x = W_new / float(W)
        scale_y = H_new / float(H)

        # Convert from np.float32 [0, 255] to PIL.Image to allow for better resizing quality.
        rgbs_out = [PIL.Image.fromarray(rgb.astype(np.uint8), mode=guess_mode(rgb)) for rgb in rgbs]

        # Resize.
        rgbs_out = [rgb.resize(size=(W_new, H_new), resample=Resampling.LANCZOS) for rgb in rgbs_out]

        # Convert back to np.float32.
        rgbs_out = [np.array(rgb).astype(np.float32) for rgb in rgbs_out]

        trajs_out = trajs.copy()
        trajs_out[..., 0] *= scale_x
        trajs_out[..., 1] *= scale_y

        return rgbs_out, trajs_out

    def _make_paths(
        self,
    ) -> Dict[str, List[Tuple[List[str], List[str], List[str], List[str], str]]]:
        """Return a dict containing relevant filepaths for the train, val, and test sets.

        The dict contains three lists, one for each train, val, and test split, containing each split's full sequence
        information. Each split's list consists of a tuple per sequence in the split, each tuple containing the image
        paths for the sequence, its trajs_2d paths, its valids paths, its visibs paths, and the sequence name.

        ```python
        sequences = {
            "train": [
                (
                    [
                        "train/path/to/ani/rgbs/rgb_00000.jpg",
                        "train/path/to/ani/rgbs/rgb_00001.jpg",
                        ...
                    ],
                    [
                        "train/path/to/ani/trajs_2d/traj_2d_00000.npy",
                        "train/path/to/ani/trajs_2d/traj_2d_00001.npy",
                        ...
                    ],
                    [
                        "train/path/to/ani/valids/valid_00000.npy",
                        "train/path/to/ani/valids/valid_00001.npy",
                        ...
                    ],
                    [
                        "train/path/to/ani/visibs/visib_00000.npy",
                        "train/path/to/ani/visibs/visib_00001.npy",
                        ...
                    ],
                    "ani",
                ),
            ],
            "test": [
                (
                    [
                        "test/path/to/ani/rgbs/rgb_00000.jpg",
                        "test/path/to/ani/rgbs/rgb_00001.jpg",
                        ...
                    ],
                    [
                        "test/path/to/ani/trajs_2d/traj_2d_00000.npy",
                        "test/path/to/ani/trajs_2d/traj_2d_00001.npy",
                        ...
                    ],
                    [
                        "test/path/to/ani/valids/valid_00000.npy",
                        "test/path/to/ani/valids/valid_00001.npy",
                        ...
                    ],
                    [
                        "test/path/to/ani/visibs/visib_00000.npy",
                        "test/path/to/ani/visibs/visib_00001.npy",
                        ...
                    ],
                    "ani",
                ),
            ],
            "val": [
                (
                    [
                        "val/path/to/ani/rgbs/rgb_00000.jpg",
                        "val/path/to/ani/rgbs/rgb_00001.jpg",
                        ...
                    ],
                    [
                        "val/path/to/ani/trajs_2d/traj_2d_00000.npy",
                        "val/path/to/ani/trajs_2d/traj_2d_00001.npy",
                        ...
                    ],
                    [
                        "val/path/to/ani/valids/valid_00000.npy",
                        "val/path/to/ani/valids/valid_00001.npy",
                        ...
                    ],
                    [
                        "val/path/to/ani/visibs/visib_00000.npy",
                        "val/path/to/ani/visibs/visib_00001.npy",
                        ...
                    ],
                    "ani",
                ),
            ]
        }
        ```

        :return: A dict containing relevant filepaths for the train, val, and test sets.
        """
        print("Building full sequences for the train, val, and test sets.")

        train_sequences = sorted([seq for seq in self.train_path.iterdir() if seq.is_dir()])
        test_sequences = sorted([seq for seq in self.test_path.iterdir() if seq.is_dir()])
        val_sequences = sorted([seq for seq in self.val_path.iterdir() if seq.is_dir()])

        sequences = {
            "train": [],  # ([rgb_paths], [traj_2d_paths], [valid_paths], [visib_paths], seq_name)
            "test": [],  # ([rgb_paths], [traj_2d_paths], [valid_paths], [visib_paths], seq_name)
            "val": [],  # ([rgb_paths], [traj_2d_paths], [valid_paths], [visib_paths], seq_name)
        }

        for seq_path in train_sequences:
            seq_name = seq_path.name
            rgb_paths = sorted(
                [str(rgb_path) for rgb_path in (seq_path / "rgbs").iterdir() if self._is_valid_file(str(rgb_path))]
            )
            traj_2d_paths = sorted(
                [
                    str(traj_2d_path)
                    for traj_2d_path in (seq_path / "trajs_2d").iterdir()
                    if self._is_valid_file(str(traj_2d_path))
                ]
            )
            valid_paths = sorted(
                [
                    str(valid_path)
                    for valid_path in (seq_path / "valids").iterdir()
                    if self._is_valid_file(str(valid_path))
                ]
            )
            visib_paths = sorted(
                [
                    str(visib_path)
                    for visib_path in (seq_path / "visibs").iterdir()
                    if self._is_valid_file(str(visib_path))
                ]
            )
            rgb_paths = rgb_paths[: self.first_n_frames] if self.first_n_frames > 0 else rgb_paths
            traj_2d_paths = traj_2d_paths[: self.first_n_frames] if self.first_n_frames > 0 else traj_2d_paths
            valid_paths = valid_paths[: self.first_n_frames] if self.first_n_frames > 0 else valid_paths
            visib_paths = visib_paths[: self.first_n_frames] if self.first_n_frames > 0 else visib_paths
            sequences["train"].append(
                (
                    rgb_paths,
                    traj_2d_paths,
                    valid_paths,
                    visib_paths,
                    seq_name,
                )
            )

        for seq_path in test_sequences:
            seq_name = seq_path.name
            rgb_paths = sorted(
                [str(rgb_path) for rgb_path in (seq_path / "rgbs").iterdir() if self._is_valid_file(str(rgb_path))]
            )
            traj_2d_paths = sorted(
                [
                    str(traj_2d_path)
                    for traj_2d_path in (seq_path / "trajs_2d").iterdir()
                    if self._is_valid_file(str(traj_2d_path))
                ]
            )
            valid_paths = sorted(
                [
                    str(valid_path)
                    for valid_path in (seq_path / "valids").iterdir()
                    if self._is_valid_file(str(valid_path))
                ]
            )
            visib_paths = sorted(
                [
                    str(visib_path)
                    for visib_path in (seq_path / "visibs").iterdir()
                    if self._is_valid_file(str(visib_path))
                ]
            )
            rgb_paths = rgb_paths[: self.first_n_frames] if self.first_n_frames > 0 else rgb_paths
            traj_2d_paths = traj_2d_paths[: self.first_n_frames] if self.first_n_frames > 0 else traj_2d_paths
            valid_paths = valid_paths[: self.first_n_frames] if self.first_n_frames > 0 else valid_paths
            visib_paths = visib_paths[: self.first_n_frames] if self.first_n_frames > 0 else visib_paths
            sequences["test"].append(
                (
                    rgb_paths,
                    traj_2d_paths,
                    valid_paths,
                    visib_paths,
                    seq_name,
                )
            )

        for seq_path in val_sequences:
            seq_name = seq_path.name
            rgb_paths = sorted(
                [str(rgb_path) for rgb_path in (seq_path / "rgbs").iterdir() if self._is_valid_file(str(rgb_path))]
            )
            traj_2d_paths = sorted(
                [
                    str(traj_2d_path)
                    for traj_2d_path in (seq_path / "trajs_2d").iterdir()
                    if self._is_valid_file(str(traj_2d_path))
                ]
            )
            valid_paths = sorted(
                [
                    str(valid_path)
                    for valid_path in (seq_path / "valids").iterdir()
                    if self._is_valid_file(str(valid_path))
                ]
            )
            visib_paths = sorted(
                [
                    str(visib_path)
                    for visib_path in (seq_path / "visibs").iterdir()
                    if self._is_valid_file(str(visib_path))
                ]
            )
            rgb_paths = rgb_paths[: self.first_n_frames] if self.first_n_frames > 0 else rgb_paths
            traj_2d_paths = traj_2d_paths[: self.first_n_frames] if self.first_n_frames > 0 else traj_2d_paths
            valid_paths = valid_paths[: self.first_n_frames] if self.first_n_frames > 0 else valid_paths
            visib_paths = visib_paths[: self.first_n_frames] if self.first_n_frames > 0 else visib_paths
            sequences["val"].append(
                (
                    rgb_paths,
                    traj_2d_paths,
                    valid_paths,
                    visib_paths,
                    seq_name,
                )
            )

        return sequences

    def _is_valid_file(self, filename: str) -> bool:
        """Checks if a file is an allowed extension.

        :param filename: Path to a file.

        :return: `True` if the filename ends with one of the extensions listed in `IMG_EXTENSIONS`.
        """
        return filename.lower().endswith(
            self.IMG_EXTENSIONS if isinstance(self.IMG_EXTENSIONS, str) else tuple(self.IMG_EXTENSIONS)
        )

    def _load_file(self, filepath: str, md5: Optional[str] = None) -> Dict[str, Any]:
        """Loads an object saved with `torch.save()` from a file.

        :param filepath: Path to the file.
        :param md5: MD5 checksum of the file. Default is `None`.

        :raises FileNotFoundError: If the file is not present or is corrupted.

        :return: A dictionary containing the file's contents.
        """
        if check_integrity(filepath, md5):
            return torch.load(filepath)
        else:
            raise FileNotFoundError(f"The meta file {filepath} is not present or is corrupted.")
