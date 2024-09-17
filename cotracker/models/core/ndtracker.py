import math
import os
from functools import partial, reduce, update_wrapper
from typing import Any, Dict, List, Optional, Tuple, Sequence
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from torch import nn


torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
torch._dynamo.config.suppress_errors = True


def exists(x: Any) -> bool:
    """Checks if object exists.

    :param x: Object to check.

    :return: `True` if object exists, `False` otherwise.
    """
    return x is not None


def default(val: Any, d: Any) -> Any:
    """Returns default value if object does not exist.

    :param val: Object to check.
    :param d: Default value.
    :return: Object if it exists, default value otherwise.
    """
    if exists(val):
        return val
    elif isfunction(d):
        return d()
    else:
        return d


def sincos_pos_emb_xy(
    xy: torch.Tensor, dim: int, max_wavelength: float = 10.0, include_xy: bool = False
) -> torch.Tensor:
    """Sinusoidal position embedding of input xy coordinates.

    Based on the position embedding technique used in the original Transformer paper (Vaswani et al., 2017).
    https://arxiv.org/abs/1706.03762

    :param xy: A tensor of xy coordinates of shape `[..., 2]`.
    :param dim: Desired embedding dimensionality. Must be an even integer.
    :param max_wavelength: Maximum wavelength of embedding. Default is `10.0`.
    :param include_xy: Whether to concatenate the original coordinates to the embedding. Default is `False`.

    :raises ValueError: If `dim` is not divisible by 4.
    :raises ValueError: If `max_wavelength` is less than or equal to 1.0.

    :return: A sinusoidal position embedding of the input xy coordinates of shape `[..., dim]` or `[..., dim + 2]` if
    `include_xy=True`.
    """
    assert xy.shape[-1] == 2

    if (dim % 4) != 0:
        raise ValueError("dim must be divisible by 4.")

    if max_wavelength <= 1.0:
        raise ValueError("max_wavelength must be greater than 1.0.")

    quarter_dim = dim // 4
    embed_term = (
        2
        * math.pi
        * torch.exp(
            torch.arange(quarter_dim, dtype=torch.float64, device=xy.device)
            * (-math.log(max_wavelength) / (quarter_dim - 1))
        )
    )

    pe_x = einsum(xy[..., 0], embed_term, "..., E -> ... E")
    pe_y = einsum(xy[..., 1], embed_term, "..., E -> ... E")

    pe = torch.cat([pe_x.sin(), pe_x.cos(), pe_y.sin(), pe_y.cos()], dim=-1)  # [..., dim]

    if include_xy:
        pe = torch.cat([xy, pe], dim=-1)  # [..., dim + 2]

    return pe.to(xy.dtype)


def xy_meshgrid(
    size: Tuple[int, int] | int,
    x_range: Tuple[float, float] | float | int = (-1.0, 1.0),
    y_range: Tuple[float, float] | float | int = (-1.0, 1.0),
    batch_size: Optional[int] = None,
    device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    """Generates a meshgrid of xy coordinates.

    :param size: A tuple `(height, width)` specifying the size of the meshgrid. Can also provide a single value `size`
    to specify the size as `(size, size)`.
    :param x_range: A tuple `(minval_x, maxval_x)` specifying the range of the x coordinates. Default is `(-1.0, 1.0)`.
    Can also provide a single value `radius` to specify the range as `(-radius, radius)`.
    :param y_range: A tuple `(minval_y, maxval_y)` specifying the range of the y coordinates. Default is `(-1.0, 1.0)`.
    Can also provide a single value `radius` to specify the range as `(-radius, radius)`.
    :param batch_size: Batch size of the meshgrid. Default is ``None``.
    :param device: Device to use. Default is ``None``.

    :return: A 2D meshgrid of xy coordinates of shape [B, H, W, C].
    """
    if isinstance(size, int):
        size = (size, size)
    if isinstance(x_range, float) or isinstance(x_range, int):
        x_range = (-x_range, x_range)
    if isinstance(y_range, float) or isinstance(y_range, int):
        y_range = (-y_range, y_range)
    x_coords, y_coords = torch.meshgrid(
        torch.linspace(x_range[0], x_range[1], size[1], device=device),
        torch.linspace(y_range[0], y_range[1], size[0], device=device),
        indexing="xy",
    )
    xy_coords = torch.stack([x_coords, y_coords], -1)  # [height, width, 2]
    if exists(batch_size):
        xy_coords = repeat(xy_coords, "h w c -> b h w c", b=batch_size)
    return xy_coords  # +x right, +y down, xy-indexed


def sample_features_5d(
    input: torch.Tensor, coords: torch.Tensor, padding_mode: str = "border", normalize_coords: bool = True
) -> torch.Tensor:
    """Sample spatio-temporal features.

    `sample_features_5d(input, coords)` works in the same way as `sample_features_4d` but for spatio-temporal features
    and points.

    :param input: A `[B, T, C, H, W]` tensor of spatio-temporal features.
    :param coords: A `[B, R1, R2, 3]` tensor of spatio-temporal points `(t, x, y)`.
    :param padding_mode: Padding mode. Defaults to `"border"`.
    :param normalize_coords: If `True`, the coordinates are normalized to `[-1, 1]`. Defaults to `True`.

    :return: A `[B, R1, R2, C]` tensor of sampled features.
    """
    # B T C H W -> B C T H W
    input = rearrange(input, "B T C H W -> B C T H W")

    # B R1 R2 3 -> B R1 R2 1 3 since bilinear_sampler expects a 5D tensor (typically of shape B T H W 3)
    coords = coords.unsqueeze(3)

    feats = bilinear_sampler(input, coords, padding_mode=padding_mode, normalize_coords=normalize_coords)  # B C R1 R2 1

    feats = rearrange(feats, "B C R1 R2 1 -> B R1 R2 C")

    return feats


def bilinear_sampler(
    input: torch.Tensor,
    coords: torch.Tensor,
    align_corners: bool = True,
    padding_mode: str = "border",
    normalize_coords: bool = True,
) -> torch.Tensor:
    """Sample a tensor using bilinear interpolation.

    `bilinear_sampler(input, coords)` samples a tensor `input` at coordinates `coords` using bilinear
    interpolation. It is the same as `torch.nn.functional.grid_sample()` but with a different coordinate convention.

    4D case:
    The input tensor is assumed to be of shape `[B, C, H, W]`, where `B` is the batch size, `C` is the number of
    channels, `H` is the height of the image, and `W` is the width of the image.

    The tensor `coords` of shape `[B, H_o, W_o, 2]` is interpreted as a meshgrid of 2D point coordinates (x, y).

    5D case:
    The input tensor is assumed to be of shape `[B, C, T, H, W]`, where `B` is the batch size, `C` is the number of
    channels, `T` is the number of images, `H` is the height of the image, and `W` is the width of the image.

    The tensor `coords` of shape `[B, T, H_o, W_o, 3]` is interpreted as a meshvolume of 3D point coordinates
    `(t, x, y)`. Note that in this case the order of the components is slightly different from `grid_sample()`, which
    would expect `(x, y, t)`.

    If `align_corners` is `True`, the coordinate `x` is assumed to be in the range `[0, W - 1]`, with `0` corresponding
    to the centre of the left-most image pixel and `W - 1` corresponding to the centre of the right-most pixel.

    If `align_corners` is `False`, the coordinate `x` is assumed to be in the range `[0, W]`, with `0` corresponding to
    the left edge of the left-most pixel and `W` corresponding to the right edge of the right-most pixel.

    Similar conventions apply to the y for the range `[0, H - 1]` and `[0, H]` and to `t` for the range `[0, T - 1]`
    and `[0, T]`.

    :param input: A `[B, C, H, W]` or `[B, C, T, H, W]` tensor of feature maps.
    :param coords: A `[B, H_o, W_o, 2]` tensor of spatial points `(x, y)` to sample the feature maps at, or a
    `[B, T, H_o, W_o, 3]` tensor of spatio-temporal points `(t, x, y)` to sample the feature volume at.
    :param align_corners: Coordinate convention. Defaults to `True`.
    :param padding_mode: Padding mode. Defaults to `"border"`.
    :param normalize_coords: If `True`, the coordinates are normalized to `[-1, 1]`. Defaults to `True`.

    :raises ValueError: If the input tensor is not 4D or 5D and if the coordinates tensor does not match the input
    dimensionality.

    :return: A `[B, C, H_o, W_o]` tensor of sampled features from feature maps or a `[B, C, T, H_o, W_o]` tensor of
    sampled features from feature volumes.
    """
    if input.ndim not in [4, 5]:
        raise ValueError("input must be 4D or 5D.")

    if input.ndim == 4 and not coords.ndim == 4:
        raise ValueError("input is 4D, but coords is not 4D.")

    if input.ndim == 5 and not coords.ndim == 5:
        raise ValueError("input is 5D, but coords is not 5D.")

    if coords.ndim == 5:
        # t x y -> x y t to match what grid_sample() expects.
        coords = coords[..., [1, 2, 0]]

    if normalize_coords:
        if align_corners:
            # Normalize coordinates from [0, W/H - 1] to [-1, 1].
            coords = (
                coords
                * torch.tensor([2 / max(size - 1, 1) for size in reversed(input.shape[2:])], device=coords.device)
                - 1
            )
        else:
            # Normalize coordinates from [0, W/H] to [-1, 1].
            coords = coords * torch.tensor([2 / size for size in reversed(input.shape[2:])], device=coords.device) - 1

    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)


class NDTracker(nn.Module):
    # TODO: update docstrings
    def __init__(
        self,
        encoder_stride: int = 4,
        encoder_dim: int = 128,
        encoder_ckpt: Optional[str] = None,
        finetune_encoder: bool = False,
        transformer_token_dim: int = 256,
        transformer_num_heads: int = 8,
        transformer_mlp_hidden_dim_mult: int = 4,
        transformer_depth: int = 6,
        transformer_dropout: float = 0.0,
        transformer_temporal_attn_dropout: float = 0.0,
        transformer_spatial_attn_dropout: float = 0.0,
        transformer_mapping_dropout: float = 0.0,
    ) -> None:
        """Initialize a NDTracker `nn.Module`.

        :param encoder_stride: The stride of the input encoder. Default is `4`.
        :param encoder_dim: The number of channels of the encoded features. Default is `128`.
        :param encoder_ckpt: The checkpoint path to load the encoder weights from. Default is `None`.
        :param finetune_encoder: Whether to finetune the encoder. Default is `False`.
        :param transformer_token_dim: The dimension of transformer tokens. Default is `256`.
        :param transformer_num_heads: The number of attention heads in the transformer. Default is `8`.
        :param transformer_mlp_hidden_dim_mult: The hidden dimension multiplier for the MLP in attention blocks.
        Default is `4`.
        :param transformer_temporal_depth: The number of temporal attention blocks in the transformer. Default is `6`.
        :param transformer_dropout: The dropout rate for the transformer network, sans Mapping network and attention
        dropout. Default is `0.0`.
        :param transformer_temporal_attn_dropout: The dropout rate for the spatial attention layers in the
        transformer network. Default is `0.0`.
        :param transformer_spatial_attn_dropout: The dropout rate for the temporal attention layers in the transformer
        network. Default is `0.0`.
        :param transformer_mapping_dropout: The dropout rate for the mapping network in the transformer. Default is
        `0.0`.
        """
        super().__init__()

        self.encoder_stride = encoder_stride
        self.encoder_dim = encoder_dim
        self.finetune_encoder = finetune_encoder
        self.transformer_token_dim = transformer_token_dim
        self.transformer_num_heads = transformer_num_heads
        self.transformer_mlp_hidden_dim_mult = transformer_mlp_hidden_dim_mult
        self.transformer_depth = transformer_depth

        # Other model properties
        self.fourier_proj_dim = 24
        num_levels = 4
        diameter = 7
        self.crop_patch_dim = diameter**2 * num_levels

        # Networks
        self.input_encoder = RAFTEncoder(in_dim=3, out_dim=self.encoder_dim, stride=self.encoder_stride)
        if exists(encoder_ckpt):
            try:
                ckpt = torch.load(encoder_ckpt)
                encoder_weights = {
                    k.removeprefix("encoder."): v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")
                }
                self.input_encoder.load_state_dict(encoder_weights)
            except Exception as e:
                print(f"Failed to load encoder weights from checkpoint: {encoder_ckpt}. Error: {e}")
        if self.finetune_encoder:
            self.input_encoder.requires_grad_(True)
        else:
            self.input_encoder.requires_grad_(False)
            self.input_encoder.eval()
        self.transformer = Transformer(
            num_layers=self.transformer_depth,
            in_dim=2 + 2 * self.fourier_proj_dim,
            input_cond_dim=self.encoder_dim + self.crop_patch_dim,
            context_dim=2 + 2 * self.fourier_proj_dim,
            # context_cond_dim=self.encoder_dim + self.crop_patch_dim,
            context_cond_dim=self.crop_patch_dim,
            hidden_dim=self.transformer_token_dim,
            num_heads=self.transformer_num_heads,
            out_dim=2,
            mlp_hidden_dim_mult=self.transformer_mlp_hidden_dim_mult,
            dropout=transformer_dropout,
            temporal_attn_dropout=transformer_temporal_attn_dropout,
            spatial_attn_dropout=transformer_spatial_attn_dropout,
            mapping_dropout=transformer_mapping_dropout,
        )

    @property
    def device(self) -> torch.device:
        """Return the device on which this module is currently stored.

        :return: The device on which this module is currently stored.
        """
        return next(self.parameters()).device

    def param_groups(self, base_lr: float = 1e-4, mapping_lr_scale: float = 1 / 3) -> List[Dict[str, Any]]:
        """Return the parameter groups for the optimizer.

        :param base_lr: The base learning rate to use for the model. Default is `1e-4`.
        :param mapping_lr_scale: The scaling factor to apply to the learning rate of the mapping network. Default is
            `1/3`.
        :return: A list of parameter groups for the optimizer.
        """
        transformer_groups = self.transformer.param_groups(base_lr=base_lr, mapping_lr_scale=mapping_lr_scale)
        encoder_groups = self.input_encoder.param_groups(base_lr=base_lr)
        groups = transformer_groups + encoder_groups
        return groups

    def forward(self, video: torch.Tensor, queries: torch.Tensor, iters: int) -> Any:
        B, T, C, H, W = video.shape
        B, N, _ = queries.shape

        self.window_size = 16

        window_size = self.window_size

        if window_size > T:
            window_size = T

        sliding_window_step_size = window_size // 2  # How much the sliding window moves at every step.
        num_windows = (T - window_size + sliding_window_step_size - 1) // sliding_window_step_size + 1

        pad = (window_size - T % window_size) % window_size
        video = F.pad(video.reshape(B, 1, T, C * H * W), (0, 0, 0, pad), "replicate").reshape(
            B, -1, C, H, W
        )
        T_trimmed = T
        B, T, C, H, W = video.shape

        # Queries will be the coordinates at the first frame of the current window.
        queried_coords = queries[..., 1:]
        queried_frames = queries[..., :1]
        queries = torch.cat([queried_coords, queried_frames], dim=2)  # [1, N, 3]

        # Initialize final predictions.
        trajs_pred_all = queried_coords[:, None].repeat(1, T, 1, 1)  # [1, T, N, 2]

        cur_frame = 0
        end_frame = cur_frame + window_size
        done = False
        query_feats = None
        trajs_prev = None
        prev_feat_corrs = None
        while not done:
            prev_end_frame = end_frame
            end_frame = cur_frame + window_size

            if end_frame > T:
                diff = end_frame - T
                end_frame = end_frame - diff
                cur_frame = max(cur_frame - diff, 0)
            frame_overlap = prev_end_frame - cur_frame if cur_frame > 0 else 1

            # print(f"Processing frames {cur_frame}:{end_frame - 1} / {T - 1}.")

            # Sample trajectories for validation losses.
            trajs_pred, query_feats, prev_feat_corrs, corrs_pyramid = self._forward(
                video=video[:, cur_frame:end_frame],
                queries=queries,
                prev_trajs_coords=trajs_prev,
                prev_feat_corrs=prev_feat_corrs,
                frame_overlap=frame_overlap,
                iters=iters,
                return_all=True,
                query_feats=query_feats,
            )

            # Update full-sequence predictions with current window predictions.
            trajs_pred_all[:, cur_frame:end_frame] = trajs_pred[:, -1]  # [1, T_window, N, 2]
            trajs_prev = trajs_pred[:, -1]  # [1, T_window, N, 2]

            if end_frame >= T:
                done = True
                trajs_pred_all = trajs_pred_all[:, :T_trimmed]
            else:
                cur_frame = cur_frame + sliding_window_step_size

        return trajs_pred_all, None, None

    # TODO: docstring
    def _forward(
        self,
        video: torch.Tensor,  # [B, T, C, H, W]
        queries: torch.Tensor,  # [B, N, 3]
        prev_trajs_coords: torch.Tensor | None,  # [B, T_prev, N, 2]
        prev_feat_corrs: torch.Tensor | None,  # [B, T_prev, N, LDD]
        frame_overlap: int,
        iters: int,
        return_all: bool = False,
        query_feats: Optional[torch.Tensor] = None,  # [B, 1, N, D]
        valids: Optional[torch.Tensor] = None,  # [B, T, N]
        **model_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        B, T, C, H, W = video.shape

        trajs, prev_trajs, query_coords, query_feats, prev_feat_corrs, corrs_pyramid = self.init(
            video=video,
            queries=queries,
            prev_trajs_coords=prev_trajs_coords,
            prev_feat_corrs=prev_feat_corrs,
            frame_overlap=frame_overlap,
            query_feats=query_feats,
        )

        # Sample trajectories through reverse diffusion.
        all_outputs = []
        for step in range(iters):
            trajs, prev_trajs, query_coords, query_feats = self.update(
                trajs=trajs,
                prev_trajs=prev_trajs,
                prev_feat_corrs=prev_feat_corrs,
                query_coords=query_coords,
                query_feats=query_feats,
                corrs_pyramid=corrs_pyramid,
                height=H,
                width=W,
                valids=valids,
                **model_kwargs,
            )

            # Rescale trajectories from [-1, 1] to [-64, W/H + 64].
            # TODO: This is a hacky way to rescale the trajectories. Find a way that's dataset agnostic.
            trajs_coords = (trajs[..., :2] + 1) * 0.5  # Rescale to [0, 1]
            trajs_coords = trajs_coords * torch.tensor([[[[W + 127, H + 127]]]], device=trajs.device) - 64

            if return_all:
                all_outputs.append(trajs_coords)

        feat_corrs = sample_corrs(
            corrs_pyramid=corrs_pyramid,
            coords=trajs_coords / self.encoder_stride,
            radius=3,
        )  # B T N LDD

        if return_all:
            # Stack outputs to [B, I, ...] tensors, where the `I` dim indexes through the refinement steps.
            all_outputs = torch.stack(all_outputs, dim=1)
        else:
            all_outputs = trajs_coords

        return all_outputs, query_feats, feat_corrs, corrs_pyramid

    # TODO: docstring
    def update(
        self,
        trajs: torch.Tensor,  # [B, T, N, 3]
        prev_trajs: torch.Tensor,  # [B, T_prev, N, 3]
        prev_feat_corrs: torch.Tensor,  # [B, T_prev, N, LDD]
        query_coords: torch.Tensor,  # [B, T, N, 2]
        query_feats: torch.Tensor,  # [B, 1, N, D]
        corrs_pyramid: List[torch.Tensor],  # [B, S, N, H // 2^level, W // 2^level]
        height: int,
        width: int,
        valids: Optional[torch.Tensor] = None,  # [B, T, N]
        **model_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trajs_coords = trajs[..., :2].detach()  # B T N 2
        trajs_frame_inds = trajs[..., 2:]  # B T N 1
        prev_trajs_coords = prev_trajs[..., :2]  # [B, T_prev, N, 2]
        prev_trajs_frame_inds = prev_trajs[..., 2:]  # [B, T_prev, N, 1]

        T = trajs_coords.shape[1]
        T_prev = prev_trajs_coords.shape[1]

        # Rescale the track coordinates from [-1, 1] to feature space pixel coordinates in order to sample from the
        # correlation maps.
        # TODO: This is a hacky way to rescale the trajectories. Find a way that's dataset agnostic.
        trajs_coords_ = (trajs_coords + 1) * 0.5
        trajs_coords_ = trajs_coords_ * torch.tensor([[[[width + 127, height + 127]]]], device=trajs.device) - 64
        trajs_coords_ = trajs_coords_ / self.encoder_stride

        # Sample correlation scores about each trajectory point.
        feat_corrs = sample_corrs(
            corrs_pyramid=corrs_pyramid,
            coords=trajs_coords_,
            radius=3,
        )  # B T N LDD

        # Embed relative coordinates using positional embeddings.
        trajs_rel_coords = trajs_coords - trajs_coords[:, :1]  # B T N 2
        trajs_rel_coords_emb = sincos_pos_emb_xy(trajs_rel_coords, self.fourier_proj_dim * 2)  # B T N 2E
        prev_trajs_rel_coords = prev_trajs_coords - trajs_coords[:, :1]  # B T_prev N 2
        prev_trajs_rel_coords_emb = sincos_pos_emb_xy(prev_trajs_rel_coords, self.fourier_proj_dim * 2)  # B T_prev N 2E

        # Prepare transformer input, conditioning, and context.
        transformer_input = torch.cat([trajs_rel_coords, trajs_rel_coords_emb], dim=-1)  # B T N (2E + 2)
        input_cond = torch.cat([repeat(query_feats, "B 1 N D -> B T N D", T=T), feat_corrs], dim=-1)  # B T N (D + LDD)
        context = torch.cat([prev_trajs_rel_coords, prev_trajs_rel_coords_emb], dim=-1)  # B T_prev N (2E + 2)
        # context_cond = torch.cat(
        #     [repeat(query_feats, "B 1 N D -> B T N D", T=T_prev), prev_feat_corrs], dim=-1
        # )  # B T_prev N (D + LDD)
        context_cond = prev_feat_corrs  # B T_prev N LDD
        # attn_mask = valids.bool() if exists(valids) else valids
        attn_mask = None

        # Compute diffusion residual prediction.
        delta_pred = self.transformer(
            input=transformer_input,
            input_pos=trajs_frame_inds,
            input_cond=input_cond,
            context=context,
            context_pos=prev_trajs_frame_inds,
            context_cond=context_cond,
            attn_mask=attn_mask,
        )  # B T N 2

        # Update trajectory coords.
        trajs_coords = trajs_coords - delta_pred

        trajs = torch.cat([trajs_coords, trajs_frame_inds], dim=-1)  # B T N 3

        return trajs, prev_trajs, query_coords, query_feats

    # TODO: docstring
    def init(
        self,
        video: torch.Tensor,  # B T C H W
        queries: torch.Tensor,  # B N 3
        prev_trajs_coords: torch.Tensor | None,  # B T_prev N 2
        prev_feat_corrs: torch.Tensor | None,  # B T_prev N LDD
        frame_overlap: int = 1,
        query_feats: Optional[torch.Tensor] = None,  # B 1 N D
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        B, T, C, H, W = video.shape
        B, N, _ = queries.shape

        assert T >= 2  # A tracker needs at least two frames to track something.
        video = 2 * (video / 255.0) - 1.0  # Rescale to [-1, 1].

        # Compute spatial (i.e., 2D) convolutional features for the video.
        # (B * T) D (H / encoder_stride) (W / encoder_stride)
        fmaps = self.input_encoder(rearrange(video, "B T C H W -> (B T) C H W"))

        # Reshape the feature maps to be [B, T, D, H / encoder_stride, W / encoder_stride].
        fmaps = rearrange(fmaps, "(B T) D h w -> B T D h w", B=B, T=T)
        D = fmaps.shape[-3]

        queried_coords = queries[..., :2]  # B N 2
        queried_frames = queries[..., 2:]  # B N 1

        # We rescale the coordinates to the feature map space since it's downscaled.
        queried_coords = queried_coords / self.encoder_stride

        # Correlate query features with the feature maps.
        # This will be used later to sample correlation scores for each trajectory point.
        queries_ = torch.cat([queried_frames, queried_coords], dim=-1).unsqueeze(1)  # B 1 N 3
        corrs_pyramid, query_feats = get_corrs_pyramid(
            fmaps=fmaps, coords=queries_, num_levels=4, sampled_feats=query_feats
        )  # List([B, T, N, H // 2^level, W // 2^level]), [B, 1, N, D]

        assert query_feats.shape == (B, 1, N, D)

        # Assumes we're in the first window, so prev_feat_corrs and prev_trajs_coords are None.
        # We make prev_trajs_coords the query points and prev_feat_corrs their sampled correlation crops.
        if not exists(prev_trajs_coords):
            prev_trajs_coords = queries[..., :2].unsqueeze(1)  # B 1 N 2
            assert not exists(prev_feat_corrs)
            prev_feat_corrs = sample_corrs(
                corrs_pyramid=[c[:, :1] for c in corrs_pyramid],
                coords=prev_trajs_coords / self.encoder_stride,
                radius=3,
            )  # B 1 N LDD

        # Rescale previous trajectory coords to [-1, 1].
        prev_trajs_coords = (prev_trajs_coords + 64) / torch.tensor(
            [[[[W + 127, H + 127]]]], device=prev_trajs_coords.device
        )
        prev_trajs_coords = 2 * prev_trajs_coords - 1  # [B, T_prev, N, 2]
        prev_trajs_frame_inds = torch.arange(
            -(prev_trajs_coords.shape[1] - 1), 1, device=self.device, dtype=prev_trajs_coords.dtype
        )
        prev_trajs_frame_inds += frame_overlap - 1
        prev_trajs_frame_inds = repeat(prev_trajs_frame_inds / (T - 1), "T_prev -> B T_prev N 1", B=B, N=N)
        prev_trajs = torch.cat([prev_trajs_coords, prev_trajs_frame_inds], dim=-1).detach()  # [B, T_prev, N, 3]

        if prev_trajs.shape[1] == 1:
            # Initialize trajectories with the last point of the previous trajectory.
            # This assumes the prev_trajs tensor has only one frame and is the query point.
            init_trajs = repeat(prev_trajs_coords[:, -1:], "B 1 ... -> B T ...", T=T)  # [B, T, N, 2]
        else:
            # Initialize the first half of the trajectories with the last half of the previous trajectory.
            # Initialize the last half of the trajectories with the last point of the previous trajectory.
            padding = prev_trajs_coords[:, -1:].repeat(1, T - frame_overlap, 1, 1)  # [B, T - overlap, N, 2]
            init_trajs = torch.cat([prev_trajs_coords[:, -frame_overlap:], padding], dim=1)  # [B, T, N, 2]
            assert init_trajs.shape[1] == T

        # Include frame indices as part of init_trajs.
        init_trajs_frame_inds = repeat(torch.arange(T, device=self.device) / (T - 1), "T -> B T N 1", B=B, N=N)
        init_trajs = torch.cat([init_trajs, init_trajs_frame_inds], dim=-1)  # [B, T, N, 3]

        # Include rescaled query coords.
        # TODO: This is a hacky way to rescale the trajectories. Find a way that's dataset agnostic.
        query_coords = (queried_coords * self.encoder_stride).unsqueeze(1)  # B 1 N 2
        query_coords = (query_coords + 64) / torch.tensor([[[[W + 127, H + 127]]]], device=query_coords.device)
        query_coords = 2 * query_coords - 1

        return init_trajs, prev_trajs, query_coords, query_feats, prev_feat_corrs, corrs_pyramid  # type: ignore


class Encoder(nn.Module):
    """Base class for encoder models."""

    def __init__(self, in_dim: int = 3, out_dim: int = 128, stride: int = 4) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride

    def _interpolate(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Perform interpolation on the input tensor.

        :param x: Input tensor.
        :param size: Size of the output tensor.
        :return: Interpolated tensor.
        """
        raise NotImplementedError("Interpolate method must be implemented in a derived class.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method must be implemented in a derived class.")


class ResidualBlock(nn.Module):
    """A residual block.

    Adapted from RAFT's implementation https://github.com/princeton-vl/RAFT/blob/master/core/extractor.py.
    """

    def __init__(self, in_dim: int, out_dim: int, norm_fn: Optional[str] = "group", stride: int = 1) -> None:
        """Initialize the ResidualBlock.

        :param in_dim: The number of input channels.
        :param out_dim: The number of output channels.
        :param norm_fn: The normalization function to use. Options are `"group"`, `"batch"`, `"instance"`, and `None`.
        Defaults to `"group"`.
        :param stride: The stride to use. Defaults to `1`.

        :raises ValueError: If `stride` is less than 1.
        :raises ValueError: If `norm_fn` is not one of `["group", "batch", "instance", None]`.
        """
        super().__init__()

        if stride < 1:
            raise ValueError("Stride must be greater than or equal to 1.")

        if norm_fn not in ["group", "batch", "instance", None]:
            raise ValueError("Invalid norm_fn. Must be one of ['group', 'batch', 'instance', None].")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm_fn = norm_fn.lower() if exists(norm_fn) else norm_fn
        self.stride = stride

        self.conv1 = nn.Conv2d(self.in_dim, self.out_dim, 3, self.stride, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)
        self.act2 = nn.ReLU(inplace=True)

        if self.norm_fn == "group":
            self.num_groups = self.out_dim // 8
            self.norm1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=self.out_dim)
            self.norm2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=self.out_dim)
            if self.stride > 1:
                self.norm3 = nn.GroupNorm(num_groups=self.num_groups, num_channels=self.out_dim)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.out_dim)
            self.norm2 = nn.BatchNorm2d(self.out_dim)
            if self.stride > 1:
                self.norm3 = nn.BatchNorm2d(self.out_dim)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.out_dim)
            self.norm2 = nn.InstanceNorm2d(self.out_dim)
            if self.stride > 1:
                self.norm3 = nn.InstanceNorm2d(self.out_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            if self.stride > 1:
                self.norm3 = nn.Identity()

        self.down = (
            nn.Sequential(
                nn.Conv2d(self.in_dim, self.out_dim, 1, self.stride),
                self.norm3,
            )
            if self.stride > 1
            else nn.Identity()
        )

        self.act3 = nn.ReLU(inplace=True)

        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self) -> None:
        """Used for initializing the weights and biases of the model."""

        def _basic_init(module) -> None:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
                if exists(module.bias):
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                if exists(module.weight):
                    nn.init.constant_(module.weight, 1)
                if exists(module.bias):
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                if exists(module.weight):
                    nn.init.constant_(module.weight, 1)
                if exists(module.bias):
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.InstanceNorm2d):
                if exists(module.weight):
                    nn.init.constant_(module.weight, 1)
                if exists(module.bias):
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResidualBlock.

        :param x: An input tensor of shape `[B, C, H, W]`.
        :return: The output tensor.
        """
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.act2(self.norm2(self.conv2(y)))

        x = self.down(x)

        out = self.act3(x + y)

        return out


class RAFTEncoder(Encoder):
    """Encoder model adapted from the BasicEncoder from RAFT
    https://github.com/princeton-vl/RAFT/blob/master/core/extractor.py.
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 128, stride: int = 4, dropout: float = 0.0) -> None:
        """Initialize the RAFTEncoder model.

        :param in_dim: Input dimension. Defaults to `3`.
        :param out_dim: Output dimension. Defaults to `128`.
        :param stride: Stride for the bilinear interpolation of encoded features. This will result in bilinearly
        interpolated features at spatial resolution `(H // stride, W // stride)`, where `(H, W)` are the height and
        width of the input features, respectively. Defaults to `4`.
        :param dropout: Dropout probability. Defaults to `0.0`.
        """
        super().__init__(in_dim=in_dim, out_dim=out_dim, stride=stride)

        self.hidden_dim_base = 32
        self.hidden_dims = [
            self.hidden_dim_base * 2,  # 64
            self.hidden_dim_base * 2,  # 64
            self.hidden_dim_base * 3,  # 96
            self.hidden_dim_base * 4,  # 128
            self.hidden_dim_base * 4,  # 128
        ]

        self.conv1 = nn.Conv2d(self.in_dim, self.hidden_dims[0], 7, stride=2, padding=3)
        self.norm1 = nn.InstanceNorm2d(self.hidden_dims[0])
        self.act1 = nn.ReLU(inplace=True)

        self.resblock1 = self._resblocks(self.hidden_dims[0], self.hidden_dims[1], 1)
        self.resblock2 = self._resblocks(self.hidden_dims[1], self.hidden_dims[2], 2)
        self.resblock3 = self._resblocks(self.hidden_dims[2], self.hidden_dims[3], 2)
        self.resblock4 = self._resblocks(self.hidden_dims[3], self.hidden_dims[4], 2)

        self.conv2 = nn.Conv2d(sum(self.hidden_dims[1:]), self.out_dim * 2, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(self.out_dim * 2)
        self.act2 = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(self.out_dim * 2, self.out_dim, 1)

        self.dropout = nn.Dropout2d(p=dropout)

        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self) -> None:
        """Used for initializing the weights and biases of the model."""

        def _basic_init(module) -> None:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if exists(module.bias):
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.InstanceNorm2d):
                if exists(module.weight):
                    nn.init.constant_(module.weight, 1)
                if exists(module.bias):
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    @staticmethod
    def _resblocks(in_dim: int, out_dim: int, stride: int = 1) -> nn.Sequential:
        """Create a sequence of residual blocks.

        :param in_dim: Input dimension.
        :param out_dim: Output dimension.
        :param stride: Stride for the first block. Defaults to `1`.

        :return: A sequence of residual blocks.
        """
        layer1 = ResidualBlock(in_dim, out_dim, norm_fn="instance", stride=stride)
        layer2 = ResidualBlock(out_dim, out_dim, norm_fn="instance", stride=1)
        return nn.Sequential(layer1, layer2)

    @staticmethod
    def _interpolate(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Perform interpolation on the input tensor.

        :param x: Input tensor of shape `[B, C, H, W]`.
        :param size: Size of the output tensor.
        :raises ValueError: If the input size is not smaller or larger than the given size for both height and width.
        :return: Interpolated tensor.
        """
        if x.shape[-2:] == size:
            return x
        if x.shape[-1:] > size[-1:] and x.shape[-2:] > size[-2:]:
            # Downsample.
            mode = "bilinear"
            align_corners = False
            antialias = True
        elif x.shape[-1:] < size[-1:] and x.shape[-2:] < size[-2:]:
            # Upsample.
            mode = "bilinear"
            align_corners = False
            antialias = True
        else:
            raise ValueError(
                f"Resizing can only occur when the given size is smaller or larger than the input size for both height and width. Input size: {x.shape[-2:]}, given size: {size}."
            )
        return F.interpolate(x, size, mode=mode, align_corners=align_corners, antialias=antialias)  # AA is slow.

    def param_groups(self, base_lr: float = 1e-4) -> List[Dict[str, Any]]:
        groups = [
            {"params": filter(lambda p: p.requires_grad, self.parameters()), "lr": base_lr},
        ]
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: Input tensor of shape `[B, C, H, W]` where `B` is batch size, `C` is the number of channels, `H` is
        height, and `W` is width.

        :return: Output tensor of shape `[B, self.dim_out, H // self.stride, W // self.stride]`.
        """
        _, _, H, W = x.shape

        y = self.act1(self.norm1(self.conv1(x)))  # [B, 64, H//2, W//2]

        a = self.resblock1(y)  # [B, 64, H//2, W//2]
        b = self.resblock2(a)  # [B, 96, H//4, W//4]
        c = self.resblock3(b)  # [B, 128, H//8, W//8]
        d = self.resblock4(c)  # [B, 128, H//16, W//16]

        new_size = (H // self.stride, W // self.stride)

        a = self._interpolate(a, size=new_size)  # [B, 64, H//4, W//4], 2x down
        b = self._interpolate(b, size=new_size)  # [B, 96, H//4, W//4], same
        c = self._interpolate(c, size=new_size)  # [B, 128, H//4, W//4], 2x up
        d = self._interpolate(d, size=new_size)  # [B, 128, H//4, W//4], 4x up

        abcd = torch.cat([a, b, c, d], dim=1)

        y = self.act2(self.norm2(self.conv2(abcd)))

        out = self.proj(y)

        out = self.dropout(out)

        return out


class compile_wrap:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._compiled_function = None
        update_wrapper(self, function)

    @property
    def compiled_function(self):
        if self._compiled_function is not None:
            return self._compiled_function
        try:
            self._compiled_function = torch.compile(self.function, *self.args, **self.kwargs)
        except RuntimeError:
            self._compiled_function = self.function
        return self._compiled_function

    def __call__(self, *args, **kwargs):
        return self.compiled_function(*args, **kwargs)


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = {tag}
    else:
        param._tags.add(tag)
    return param


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param


@compile_wrap
def _l2_norm(x: torch.Tensor, scale: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """Scales the input tensor by the given scale tensor by first normalizing the input tensor by its L2 norm then
    multiplying it by the given scale.

    :param x: The input tensor.
    :param scale: The scale tensor.
    :param dim: The dimension to sum over.
    :param eps: A small value to prevent division by zero. Defaults to `1e-6`.
    :return: The scaled input tensor.
    """
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    sum_sq = torch.sum(x.to(dtype) ** 2, dim=dim, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_x = sqrt_scale * torch.rsqrt(sum_sq + eps)
    return x * scale_x.to(x.dtype)


@compile_wrap
def _rms_norm(
    x: torch.Tensor, scale: torch.Tensor, shift: Optional[torch.Tensor], dim: int, eps: float = 1e-6
) -> torch.Tensor:
    """Scales the input tensor by the given scale tensor by first normalizing the input tensor by the square root of
    the mean squared value across dim then multiplying it by the given scale.

    :param x: The input tensor.
    :param dim: The dimension to sum over.
    :param scale: The scale tensor.
    :param shift: The shift tensor. Defaults to `None`.
    :param eps: A small value to prevent division by zero. Defaults to `1e-6`.
    :return: The normalized input tensor.
    """
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=dim, keepdim=True)
    scale_x = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    if shift is None:
        return x * scale_x.to(x.dtype)
    else:
        return x * scale_x.to(x.dtype) + shift.to(dtype)


@compile_wrap
def linear_geglu(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


@compile_wrap
def dot_product_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_dropout: float = 0.0,
    return_attn_weights: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Computes the dot-product attention.

    :param q: The query tensor.
    :param k: The key tensor.
    :param v: The value tensor.
    :param attn_mask: An optional attention mask tensor. Defaults to `None`.
    :param attn_dropout: The dropout rate for the attention weights, post-softmax. Defaults to `0.0`.
    :param return_attn_weights: Whether to return the attention weights. Defaults to `False`.

    :return: The attention tensor and attention weights if requested.
    """
    attn_weights = q @ k.transpose(-2, -1)
    if exists(attn_mask):
        max_neg_value = -torch.finfo(q.dtype).max
        attn_bias = (~attn_mask) * max_neg_value
        attn_weights += attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=attn_dropout, training=True)
    out = attn_weights @ v
    if return_attn_weights:
        return out, attn_weights
    else:
        return out, None


@compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj) -> None:
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class LinearGEGLU(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__(in_dim, out_dim * 2, bias=bias)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer."""

    def __init__(
        self, shape: Sequence[int], eps: float = 1e-6, allow_shift: bool = False, learnable: bool = True
    ) -> None:
        """Initializes the RMSNorm layer.

        :param shape: The shape of the input tensor.
        :param eps: A small value to prevent division by zero. Defaults to `1e-6`.
        :param allow_shift: Whether to allow shifting. Defaults to `False`.
        :param learnable: Whether to learn the scaling and shifting. Defaults to `True`.
        """
        super().__init__()
        self.shape = shape
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(self.shape), requires_grad=learnable)
        self.shift = nn.Parameter(torch.zeros(self.shape), requires_grad=learnable) if allow_shift else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        :param x: The input tensor.
        :return: The normalized input tensor.
        """
        return _rms_norm(x, self.scale, self.shift, dim=-1, eps=self.eps)


class AdaRMSNorm(nn.Module):
    """Adaptive Root Mean Square Normalization layer."""

    def __init__(self, in_dim: int, cond_dim: int, eps: float = 1e-6, allow_shift: bool = False) -> None:
        """Initializes the AdaRMSNorm layer.

        :param in_dim: The number of input channels.
        :param cond_dim: The number of conditioning channels.
        :param eps: A small value to prevent division by zero. Defaults to `1e-6`.
        :param allow_shift: Whether to allow shifting. Defaults to `False`.
        """
        super().__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.eps = eps
        self.allow_shift = allow_shift
        linear_dim = self.in_dim * 2 if self.allow_shift else self.in_dim
        self.linear = apply_wd(zero_init(nn.Linear(self.cond_dim, linear_dim, bias=False)))
        tag_module(self.linear, "mapping")

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        :param x: The input tensor.
        :param cond: The conditioning tensor.
        :return: The normalized input tensor.
        """
        scale, shift = self.linear(cond).chunk(2, dim=-1) if self.allow_shift else (self.linear(cond), None)
        return _rms_norm(x, scale + 1.0, shift, dim=-1, eps=self.eps)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, theta: torch.Tensor, conj: torch.Tensor) -> torch.Tensor:
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs: torch.Tensor, output: torch.Tensor) -> None:
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (theta,) = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)  # type: ignore


class AxialRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_frequency: float = 10.0) -> None:
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(max_frequency * math.pi)
        freqs = torch.linspace(log_min, log_max, num_heads * dim // 2 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 2, num_heads).T.contiguous())

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        theta = pos[..., None, :] * self.freqs.to(pos.dtype)
        return theta


class Attention(nn.Module):
    """Attention module."""

    def __init__(
        self,
        in_dim: int,
        context_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        query_key_dim: int = 48,
        value_dim: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        learnable_scale: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        """Initialize the Attention module.

        :param in_dim: The number of channels for the input tensor `x`. The input tensor is used to create the
        query tensor and is also used to create the key and value tensors if `context` is not given.
        :param context_dim: The number of channels for the context tensor. Defaults to `None`. If `None`, it will be
        set to `in_dim`. The context tensor is used to create the key and value tensors if it is provided.
        :param out_dim: The number of output channels. Defaults to `None`. If `None`, it will be set to `in_dim`.
        :param query_key_dim: The number of channels for the query and key tensors. Defaults to `48`.
        :param value_dim: The number of channels for the value tensor. Defaults to `None`. If `None`, it will be set to
        `query_key_dim`.
        :param num_heads: The number of attention heads. Defaults to `8`.
        :param qkv_bias: Whether to include bias in the query, key, and value projections. Defaults to `False`.
        :param learnable_scale: Whether to learn the scaling factor applied pre-softmax for each head.
        Defaults to `True`.
        :param dropout: The dropout rate. Defaults to ``0.0``.
        :param attn_dropout: The dropout rate for the attention weights, post-softmax. Defaults to `0.0`.
        """
        super().__init__()
        self.in_dim = in_dim
        self.context_dim = default(context_dim, in_dim)
        self.out_dim = default(out_dim, in_dim)
        self.query_key_dim = query_key_dim
        self.value_dim = default(value_dim, query_key_dim)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.learnable_scale = learnable_scale
        self.attn_dropout = attn_dropout
        self.dropout = nn.Dropout(dropout, inplace=True)

        self.pos_emb = AxialRoPE(self.query_key_dim, self.num_heads)

        if self.learnable_scale:
            self.scale = nn.Parameter(torch.full([self.num_heads], 10.0))
        else:
            self.scale = torch.full([self.num_heads], 1.0)

        self.to_q = apply_wd(nn.Linear(self.in_dim, self.query_key_dim * self.num_heads, bias=self.qkv_bias))
        self.to_kv = apply_wd(
            nn.Linear(self.context_dim, (self.query_key_dim + self.value_dim) * self.num_heads, bias=self.qkv_bias)
        )

        self.to_out = apply_wd(zero_init(nn.Linear(self.value_dim * self.num_heads, self.out_dim, bias=False)))

    def forward(
        self,
        input: torch.Tensor,
        input_pos: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of Attention module.

        :param x: An input tensor from which queries, keys, and values can be created from. Expected shape is
        `[B, ..., N_query, C_query]`.
        :param context: An optional context tensor from which keys and values can be created from. Expected shape is
        `[B, ..., N_context, C_context]`. Default is `None`.
        :param attn_mask: An optional attention mask tensor. Shape must be broadcastable to the shape of the attention
        tensor, which is `[B, ..., N_query, N_query]` or `[B, ..., N_query, N_context]` if `context` is given.
        Defaults to `None`.

        :return: The transformed input tensor using attention.
        """
        q = rearrange(
            self.to_q(input), "B N (H C_query) -> B H N C_query", H=self.num_heads, C_query=self.query_key_dim
        )
        kv = self.to_kv(default(context, input))
        kv = rearrange(kv, "B N (H C_kv) -> B H N C_kv", H=self.num_heads, C_kv=self.query_key_dim + self.value_dim)
        k, v = kv[..., : self.query_key_dim], kv[..., self.query_key_dim :]

        q = _l2_norm(q, self.scale[None, :, None, None], dim=-1)
        k = _l2_norm(k, self.scale[None, :, None, None], dim=-1)

        theta_q = self.pos_emb(input_pos)
        theta_k = self.pos_emb(context_pos) if exists(context) and exists(context_pos) else theta_q
        theta_q = rearrange(theta_q, "B N H C -> B H N C")
        theta_k = rearrange(theta_k, "B N H C -> B H N C")
        q = apply_rotary_emb_(q, theta_q)
        k = apply_rotary_emb_(k, theta_k)

        out, _ = dot_product_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            attn_dropout=self.attn_dropout if self.training else 0.0,
            return_attn_weights=False,
        )

        out = rearrange(out, "B H N C_value -> B N (H C_value)")
        out = self.dropout(out)
        out = self.to_out(out)

        return out


class AttentionBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        input_cond_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        context_cond_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.input_cond_dim = input_cond_dim
        self.context_dim = context_dim
        self.context_cond_dim = context_cond_dim
        self.num_heads = num_heads

        self.norm_input = (
            AdaRMSNorm(self.in_dim, self.input_cond_dim) if exists(self.input_cond_dim) else RMSNorm([self.in_dim])
        )
        if exists(self.context_dim):
            self.norm_context = (
                AdaRMSNorm(self.context_dim, self.context_cond_dim)
                if exists(self.context_cond_dim)
                else RMSNorm([self.context_dim])
            )
        else:
            self.norm_context = nn.Identity()

        self.attn = Attention(
            self.in_dim,
            context_dim=self.context_dim,
            num_heads=self.num_heads,
            qkv_bias=False,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

    def forward(
        self,
        input: torch.Tensor,
        input_pos: torch.Tensor,
        input_cond: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_pos: Optional[torch.Tensor] = None,
        context_cond: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        skip = input
        if exists(input_cond) and exists(self.input_cond_dim):
            input = self.norm_input(input, input_cond)
        elif exists(self.input_cond_dim):
            input = _rms_norm(input, torch.ones_like(input), None, dim=-1)
        else:
            input = self.norm_input(input)

        if exists(context) and exists(self.context_dim):
            if exists(context_cond) and exists(self.context_cond_dim):
                context = self.norm_context(context, context_cond)
            elif exists(self.context_cond_dim):
                context = _rms_norm(context, torch.ones_like(context), None, dim=-1)
            else:
                context = self.norm_context(context)

        input = self.attn(input, input_pos, context=context, context_pos=context_pos, attn_mask=attn_mask)

        return input + skip


class FeedForwardBlock(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, input_cond_dim: Optional[int] = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.input_cond_dim = input_cond_dim
        self.hidden_dim = hidden_dim

        self.norm = (
            AdaRMSNorm(self.in_dim, self.input_cond_dim) if exists(self.input_cond_dim) else RMSNorm([self.in_dim])
        )
        self.up_proj = apply_wd(LinearGEGLU(self.in_dim, self.hidden_dim, bias=False))
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.down_proj = apply_wd(zero_init(nn.Linear(self.hidden_dim, self.in_dim, bias=False)))

    def forward(self, input: torch.Tensor, input_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip = input
        if exists(input_cond) and exists(self.input_cond_dim):
            input = self.norm(input, input_cond)
        elif exists(self.input_cond_dim):
            input = _rms_norm(input, torch.ones_like(input), None, dim=-1)
        else:
            input = self.norm(input)
        input = self.up_proj(input)
        input = self.dropout(input)
        input = self.down_proj(input)
        return input + skip


class GlobalTransformerLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        input_cond_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        context_cond_dim: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.input_cond_dim = input_cond_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.context_cond_dim = context_cond_dim

        self.attn = AttentionBlock(
            in_dim=self.in_dim,
            input_cond_dim=self.input_cond_dim,
            context_dim=self.context_dim,
            context_cond_dim=self.context_cond_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.ff = FeedForwardBlock(
            in_dim=self.in_dim, hidden_dim=self.hidden_dim, input_cond_dim=self.input_cond_dim, dropout=dropout
        )

    def forward(
        self,
        input: torch.Tensor,
        input_pos: torch.Tensor,
        input_cond: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_pos: Optional[torch.Tensor] = None,
        context_cond: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input = self.attn(input, input_pos, input_cond, context, context_pos, context_cond, attn_mask)
        input = self.ff(input, input_cond)
        return input


class MappingFeedForwardBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.norm = RMSNorm([self.in_dim])
        self.up_proj = apply_wd(LinearGEGLU(self.in_dim, self.hidden_dim, bias=False))
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.down_proj = apply_wd(zero_init(nn.Linear(self.hidden_dim, self.in_dim, bias=False)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.in_norm = RMSNorm([self.in_dim])
        self.blocks = nn.ModuleList(
            [MappingFeedForwardBlock(in_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        if self.out_dim != self.in_dim:
            self.proj = apply_wd(nn.Linear(self.in_dim, self.out_dim, bias=False))
        else:
            self.proj = nn.Identity()
        self.out_norm = RMSNorm([self.out_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)
        x = self.out_norm(x)
        return x


class Transformer(nn.Module):
    """Transformer model that updates track estimates."""

    def __init__(
        self,
        num_layers: int,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        out_dim: int,
        mlp_hidden_dim_mult: int,
        input_cond_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        context_cond_dim: Optional[int] = None,
        dropout: float = 0.0,
        temporal_attn_dropout: float = 0.0,
        spatial_attn_dropout: float = 0.0,
        mapping_dropout: float = 0.0,
    ) -> None:
        """Initializes the Transformer.

        :param num_layers: Number of transformer layers.
        :param in_dim: Number of channels in the input.
        :param hidden_dim: Number of hidden channels of the model, i.e., the token dimensionality.
        :param num_heads: Number of heads in the multi-head attention.
        :param out_dim: Number of channels in the the output.
        :param mlp_hidden_dim_mult: The hidden dimension multiplier for the MLP in the attention blocks.
        :param input_cond_dim: Number of channels in the input conditioning tensor. Default is `None`.
        :param context_dim: Number of channels in the context tensor. Default is `None`.
        :param context_cond_dim: Number of channels in the context conditioning tensor. Default is `None`.
        :param dropout: Dropout rate. Default is `0.0`.
        :param temporal_attn_dropout: Dropout rate for the temporal self-attention weights, post-softmax.
        Default is `0.0`.
        :param spatial_attn_dropout: Dropout rate for the spatial attention weights, post-softmax. Default is `0.0`.
        :param mapping_dropout: Dropout rate for the mapping network. Default is `0.0`.
        """
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.input_cond_dim = input_cond_dim
        self.context_dim = context_dim
        self.context_cond_dim = context_cond_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.mlp_hidden_dim_mult = mlp_hidden_dim_mult

        # Tokenizes the input.
        self.input_transform = apply_wd(nn.Linear(self.in_dim, self.hidden_dim, bias=False))

        # Tokenizes the context.
        if exists(self.context_dim):
            self.context_transform = apply_wd(nn.Linear(self.context_dim, self.hidden_dim, bias=False))

        self.virtual_points = nn.Parameter(torch.randn(1, 1, 64, self.hidden_dim))

        # Mapping network for conditioning input.
        if exists(self.input_cond_dim):
            self.input_cond_mapping = MappingNetwork(
                in_dim=self.input_cond_dim,  # type: ignore
                hidden_dim=self.hidden_dim * self.mlp_hidden_dim_mult,
                out_dim=self.input_cond_dim,  # type: ignore
                num_layers=2,
                dropout=mapping_dropout,
            )
            self.input_cond_mapping = tag_module(self.input_cond_mapping, "mapping")

        # Mapping network for conditioning context.
        if exists(self.context_cond_dim):
            self.context_cond_mapping = MappingNetwork(
                in_dim=self.context_cond_dim,  # type: ignore
                hidden_dim=self.hidden_dim * self.mlp_hidden_dim_mult,
                out_dim=self.context_cond_dim,  # type: ignore
                num_layers=2,
                dropout=mapping_dropout,
            )
            self.context_cond_mapping = tag_module(self.context_cond_mapping, "mapping")

        # Initialize the attention blocks.
        Input2ContextCrossAttnBlock = partial(
            GlobalTransformerLayer,
            in_dim=self.hidden_dim,
            input_cond_dim=self.input_cond_dim,
            hidden_dim=self.hidden_dim * self.mlp_hidden_dim_mult,
            context_dim=self.hidden_dim,
            context_cond_dim=self.context_cond_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            attn_dropout=temporal_attn_dropout,
        )
        Virtual2InputCrossAttnBlock = partial(
            GlobalTransformerLayer,
            in_dim=self.hidden_dim,
            input_cond_dim=None,
            hidden_dim=self.hidden_dim * self.mlp_hidden_dim_mult,
            context_dim=self.hidden_dim,
            context_cond_dim=self.input_cond_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            attn_dropout=spatial_attn_dropout,
        )
        Input2VirtualCrossAttnBlock = partial(
            GlobalTransformerLayer,
            in_dim=self.hidden_dim,
            input_cond_dim=self.input_cond_dim,
            hidden_dim=self.hidden_dim * self.mlp_hidden_dim_mult,
            context_dim=self.hidden_dim,
            context_cond_dim=None,
            num_heads=self.num_heads,
            dropout=dropout,
            attn_dropout=spatial_attn_dropout,
        )
        InputSelfAttnBlock = partial(
            GlobalTransformerLayer,
            in_dim=self.hidden_dim,
            input_cond_dim=self.input_cond_dim,
            hidden_dim=self.hidden_dim * self.mlp_hidden_dim_mult,
            context_dim=None,
            context_cond_dim=None,
            num_heads=self.num_heads,
            dropout=dropout,
            attn_dropout=temporal_attn_dropout,
        )

        # Transformer layers.
        if exists(self.context_dim):
            self.input2context_xattn_block = Input2ContextCrossAttnBlock()
        self.virtual2input_xattn_block = Virtual2InputCrossAttnBlock()
        self.input2virtual_xattn_block = Input2VirtualCrossAttnBlock()
        self.input_self_attn_blocks = nn.ModuleList([InputSelfAttnBlock() for _ in range(self.num_layers)])
        self.out_norm = RMSNorm([self.hidden_dim])
        self.out = apply_wd(zero_init(nn.Linear(self.hidden_dim, self.out_dim, bias=False)))

    def param_groups(self, base_lr: float = 1e-4, mapping_lr_scale: float = 1 / 3) -> List[Dict[str, Any]]:
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": filter(lambda p: p.requires_grad, list(wd)), "lr": base_lr},
            {"params": filter(lambda p: p.requires_grad, list(no_wd)), "lr": base_lr, "weight_decay": 0.0},
            {"params": filter(lambda p: p.requires_grad, list(mapping_wd)), "lr": base_lr * mapping_lr_scale},
            {
                "params": filter(lambda p: p.requires_grad, list(mapping_no_wd)),
                "lr": base_lr * mapping_lr_scale,
                "weight_decay": 0.0,
            },
        ]
        return groups

    def forward(
        self,
        input: torch.Tensor,
        input_pos: torch.Tensor,
        input_cond: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_pos: Optional[torch.Tensor] = None,
        context_cond: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model to produce track updates.

        :param input_tensor: Input tensor of shape `[B, T, N, C]` where `B` is the batch size, `T` is the number of
        frames, `N` is the number of tracks, and `C` is the number of channels.
        :param context: Context tensor of shape `[B, H, N, C]` where `H` is the number of history frames. Default is
        `None`.
        :param cond: Conditioning tensor of shape `[B, T, N, F]` where `F` is the number of features. Default is `None`.
        :param attn_mask: A boolean attention mask tensor of shape `[B, T, N]`. Mask values of `True` indicates that
        the element should take part in attention. Default is `None`.

        :return: Output tensor of shape `[B, T, N, O]` where `O` is the output dimensionality. This is used for
        updating the track estimates (the coordinates and features).
        """
        # Tokenize input.
        tokens = self.input_transform(input)
        B, T, N, E = tokens.shape
        tokens = rearrange(tokens, "B T N E -> (B N) T E")
        tokens_pos = rearrange(input_pos, "B T N E -> (B N) T E")

        context_exists = exists(context) and exists(self.context_dim)
        input_cond_exists = exists(input_cond) and exists(self.input_cond_dim)
        context_cond_exists = exists(context_cond) and exists(self.context_cond_dim)

        # Tokenize context.
        if context_exists:
            context_tokens = self.context_transform(context)
            B, H, N, E = context_tokens.shape
            context_tokens = rearrange(context_tokens, "B H N E -> (B N) H E")
            context_tokens_pos = rearrange(context_pos, "B H N E -> (B N) H E")

        if exists(attn_mask):
            if context_exists:
                input2context_attn_mask = rearrange(attn_mask, "B T N -> (B N) 1 T 1")  # [BN, num_heads, T, H]
            virtual2input_attn_mask = rearrange(attn_mask, "B T N -> (B T) 1 1 N")  # [BT, num_heads, V, N]
            input2virtual_attn_mask = rearrange(attn_mask, "B T N -> (B T) 1 N 1")  # [BT, num_heads, N, V]
            input_attn_mask = rearrange(attn_mask, "B T N -> (B N) 1 T 1")  # [BN, num_heads, T, T]
        else:
            input2context_attn_mask = virtual2input_attn_mask = input2virtual_attn_mask = input_attn_mask = None

        if input_cond_exists:
            input_cond = self.input_cond_mapping(input_cond)  # B T N F
            input_cond = rearrange(input_cond, "B T N F -> (B N) T F")

        if context_cond_exists:
            context_cond = self.context_cond_mapping(context_cond)  # B H N F
            context_cond = rearrange(context_cond, "B H N F -> (B N) H F")

        # Prepare virtual tokens.
        virtual_tokens = repeat(self.virtual_points, "1 1 V E -> B T V E", B=B, T=T)
        virtual_tokens = rearrange(virtual_tokens, "B T V E -> (B T) V E")
        virtual_tokens_pos = repeat(input_pos[..., :1, :], "B T 1 E -> B T V E", V=64)
        virtual_tokens_pos = rearrange(virtual_tokens_pos, "B T V E -> (B T) V E")

        if context_exists:
            # Update input tokens by having them temporally x-attend to context_tokens.
            tokens = self.input2context_xattn_block(
                input=tokens,
                input_pos=tokens_pos,
                input_cond=input_cond,
                context=context_tokens,
                context_pos=context_tokens_pos,
                context_cond=context_cond,
                attn_mask=input2context_attn_mask,
            )

        # Rearranging input and virtual tokens for spatial x-attention.
        tokens = rearrange(tokens, "(B N) T E -> (B T) N E", B=B, N=N)
        tokens_pos = rearrange(tokens_pos, "(B N) T E -> (B T) N E", B=B, N=N)
        input_cond = rearrange(input_cond, "(B N) T F -> (B T) N F", B=B, N=N) if input_cond_exists else input_cond

        # Update virtual tokens by having them spatially x-attend to input tokens.
        virtual_tokens = self.virtual2input_xattn_block(
            input=virtual_tokens,
            input_pos=virtual_tokens_pos,
            context=tokens,
            context_pos=tokens_pos,
            context_cond=input_cond,
            attn_mask=virtual2input_attn_mask,
        )

        # Update input tokens by having them spatially x-attend to virtual tokens.
        tokens = self.input2virtual_xattn_block(
            input=tokens,
            input_pos=tokens_pos,
            input_cond=input_cond,
            context=virtual_tokens,
            context_pos=virtual_tokens_pos,
            attn_mask=input2virtual_attn_mask,
        )

        # Rearranging input and virtual tokens for self-attention.
        tokens = rearrange(tokens, "(B T) N E -> (B N) T E", B=B, T=T)
        tokens_pos = rearrange(tokens_pos, "(B T) N E -> (B N) T E", B=B, T=T)
        input_cond = rearrange(input_cond, "(B T) N F -> (B N) T F", B=B, T=T) if input_cond_exists else input_cond

        for i in range(self.num_layers):
            # Update input tokens by having them temporally self-attend to themselves.
            tokens = self.input_self_attn_blocks[i](
                input=tokens,
                input_pos=tokens_pos,
                input_cond=input_cond,
                attn_mask=input_attn_mask,
            )

        tokens = self.out_norm(tokens)
        tokens = rearrange(tokens, "(B N) T E -> B T N E", B=B, N=N)
        out = self.out(tokens)

        return out


@compile_wrap
def sample_corrs(
    corrs_pyramid: List[torch.Tensor],
    coords: torch.Tensor,
    radius: int,
    normalize_coords: bool = True,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Sample from a correlation pyramid.

    :param corrs_pyramid: The correlation pyramid to sample from. The correlation pyramid is a list of tensors each of
    shape `[B, T, N, H // 2^level, W // 2^level]`, where `level` indicates the tensor index in the list.
    :param coords: The coordinates at which to sample the correlation pyramid from. These coordinates correspond to
    sampling at the highest resolution of the pyramid (lowest level). They are scaled according to the pyramid
    level. The coordinates form a spatiotemporal meshgrid of shape `[B, T, N, 2]`, where `B` is the batch size, `T`
    is the temporal extent, `N` is the number of coordinates, and `2` corresponds to the x and y coordinates at
    frame `t` out of the `T` frames.
    :param radius: The radius of the local receptive field about each correlation point. The local receptive field
    is a square of size `(2 * radius + 1) x (2 * radius + 1)`.
    :param normalize_coords: Whether to normalize the coordinates to the range `[-1, 1]`. Defaults to `True`.
    :param padding_mode: The padding mode to use when sampling the correlation pyramid. Defaults to `border`.

    :return: The sampled correlation pyramid. The output tensor has shape
    `[B, T, N, diameter * diameter * num_levels]`, where `diameter * diameter * num_levels` is the number of
    correlation values at each coordinate, corresponding to the local receptive field size dictated by `radius` and the
    number of pyramid levels dictated by the length of `corrs_pyramid` (`num_levels`).
    """
    B, T, N, D = coords.shape
    assert D == 2, "Coordinates must be 2D."
    assert len(corrs_pyramid) > 0, "Correlation pyramid is empty."

    # A xy-coordinate meshgrid of shape [1, 2 * radius + 1, 2 * radius + 1, 2] representing the local receptive
    # field about each correlation point.
    diameter = 2 * radius + 1
    delta = xy_meshgrid(
        size=diameter, x_range=radius, y_range=radius, batch_size=1, device=corrs_pyramid[0].device
    )  # 1, diameter, diameter, 2

    sampled_corrs_pyramid = []
    for level in range(len(corrs_pyramid)):
        corrs = corrs_pyramid[level]  # B, T, N, H // 2**level, W // 2**level
        corrs = rearrange(corrs, "b t n h w -> (b t n) 1 h w")

        centroid_level = rearrange(coords, "b t n c -> (b t n) 1 1 c") / 2**level
        coords_level = centroid_level + delta  # BTN, diameter, diameter, 2

        sampled_corrs = bilinear_sampler(
            corrs,
            coords_level,
            padding_mode=padding_mode,
            normalize_coords=normalize_coords,
        )  # BSN, 1, diameter, diameter
        sampled_corrs = rearrange(sampled_corrs, "(b t n) c h w -> b t n (c h w)", b=B, t=T, n=N)
        sampled_corrs_pyramid.append(sampled_corrs)

    out = torch.cat(sampled_corrs_pyramid, dim=-1)  # B, T, N, (diameter**2) * num_levels

    return out


@compile_wrap
def sample_feats(
    fmaps_pyramid: List[torch.Tensor],
    coords: torch.Tensor,
    radius: int,
    normalize_coords: bool = True,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Sample from a feature maps pyramid.

    :param fmaps_pyramid: The feature maps pyramid to sample from. The feature maps pyramid is a list of tensors each of
    shape `[B, T, C, H // 2^level, W // 2^level]`, where `level` indicates the tensor index in the list.
    :param coords: The coordinates at which to sample the feature maps pyramid from. These coordinates correspond to
    sampling at the highest resolution of the pyramid (lowest level). They are scaled according to the pyramid
    level. The coordinates form a spatiotemporal meshgrid of shape `[B, T, N, 2]`, where `B` is the batch size, `T`
    is the temporal extent, `N` is the number of coordinates, and `2` corresponds to the x and y coordinates at
    frame `t` out of the `T` frames.
    :param radius: The radius of the local receptive field about each correlation point. The local receptive field
    is a square of size `(2 * radius + 1) x (2 * radius + 1)`.
    :param normalize_coords: Whether to normalize the coordinates to the range `[-1, 1]`. Defaults to `True`.
    :param padding_mode: The padding mode to use when sampling the feature maps pyramid. Defaults to `border`.

    :return: The sampled feature maps pyramid. The output tensor has shape
    `[B, T, N, C * diameter * diameter * num_levels]`, where `C * diameter * diameter * num_levels` is the number
    of features at each coordinate, corresponding to the number of pyramid levels dictated by the length of
    `fmaps_pyramid` (`num_levels`)
    """
    B, T, N, D = coords.shape
    assert D == 2, "Coordinates must be 2D."
    assert len(fmaps_pyramid) > 0, "Feature pyramid is empty."

    # A xy-coordinate meshgrid of shape [1, 2 * radius + 1, 2 * radius + 1, 2] representing the local receptive
    # field about each correlation point.
    diameter = 2 * radius + 1
    delta = xy_meshgrid(
        size=diameter, x_range=radius, y_range=radius, batch_size=1, device=fmaps_pyramid[0].device
    )  # 1, diameter, diameter, 2
    delta = rearrange(delta, "1 h w c -> 1 1 h w c")  # 1, 1, diameter, diameter, 2

    sampled_feats_pyramid = []
    for level in range(len(fmaps_pyramid)):
        feats = fmaps_pyramid[level]  # B, T, C, H // 2**level, W // 2**level
        feats = rearrange(feats, "b t c h w -> (b t) c h w")

        centroid_level = rearrange(coords, "b t n c -> (b t) n 1 1 c") / 2**level
        coords_level = centroid_level + delta  # BT, N, diameter, diameter, 2

        def _bilinear_sampler(coords_: torch.Tensor) -> torch.Tensor:
            return bilinear_sampler(
                feats,  # BT, C, H // 2**level, W // 2**level
                coords_,  # BT, diameter, diameter, 2
                padding_mode=padding_mode,
                normalize_coords=normalize_coords,
            )

        batched_bilinear_sampler = torch.vmap(_bilinear_sampler, in_dims=1, out_dims=1)

        sampled_feats = batched_bilinear_sampler(coords_level)  # BT, N, C, diameter, diameter
        sampled_feats = rearrange(sampled_feats, "(b t) n c h w -> b t n (c h w)", b=B, t=T)
        sampled_feats_pyramid.append(sampled_feats)

    out = torch.cat(sampled_feats_pyramid, dim=-1)  # B, T, N, (C * diameter**2) * num_levels

    return out


@compile_wrap
def get_corrs_pyramid(
    fmaps: torch.Tensor,
    coords: torch.Tensor,
    num_levels: int,
    sampled_feats: Optional[torch.Tensor] = None,
    padding_mode: str = "border",
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Compute a correlation pyramid.

    This returns a list of `[B, T, N, H // 2^level, W // 2^level]` tensors representing correlations between `fmaps`
    and features located at `coords` within `fmaps`. Each pyramid level is downsampled by a factor of `2^level`, where
    `level` is the level of the pyramid, starting from 0 at the original resolution and ending at `num_levels-1` at the
    smallest resolution.

    :param fmaps: The feature maps to use for computing the correlation pyramid. They should be of shape
    `[B, T, C, H, W]`, where `B` is the batch size, `T` is the number of feature maps (corresponding to temporal
    length), `C` is the number of channels, and `H` and `W` are the height and width of the feature maps, respectively.
    :param coords: A `[B, 1, N, 3]` tensor of spatiotemporal points `(t, x, y)`, where `N` is the number of points.
    :param num_levels: The number of levels to use in the pyramid.
    :param sampled_feats: The features sampled at `coords` at the base pyramid level. If not provided, they are computed
    from `fmaps`. Defaults to `None`.
    :param padding_mode: The padding mode to use when sampling the feature maps. Defaults to `border`.

    :return: A list of `[B, T, N, H // 2^level, W // 2^level]` tensors representing the correlation pyramid and
    the features sampled at `coords` at the base pyramid level.
    """
    B, T, C, H, W = fmaps.shape
    B, R, N, D = coords.shape
    assert R == 1 and D == 3, "coords must be of shape [B, 1, N, 3]."

    norm_factor = 1.0 / torch.sqrt(torch.tensor(C))

    # Sample features at coords at the base pyramid level.
    if not exists(sampled_feats):
        sampled_feats = sample_features_5d(fmaps, coords, padding_mode=padding_mode)  # [B, 1, N, D]

    # Correlate feature maps with the sampled features.
    sampled_feats_ = repeat(sampled_feats, "B 1 N D -> B T N D", T=T)
    corrs = einsum(sampled_feats_, rearrange(fmaps, "B T C H W -> B T C (H W)"), "B T I C, B T C J -> B T I J")
    corrs = rearrange(corrs, "B T N (H W) -> B T N H W", H=H, W=W) * norm_factor

    corrs_pyramid = [corrs]
    for _ in range(num_levels - 1):
        corrs_ = rearrange(corrs, "B T N H W -> (B T) N H W")
        corrs_ = F.interpolate(corrs_, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)
        corrs = rearrange(corrs_, "(B T) N h w -> B T N h w", B=B, T=T)

        # Append the correlations to the pyramid.
        corrs_pyramid.append(corrs)

    return corrs_pyramid, sampled_feats  # type: ignore
