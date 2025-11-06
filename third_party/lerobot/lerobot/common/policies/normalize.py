#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

def _first_stat_shape(d):
    """Return shape from any of mean/std/min/max present, or None."""
    for k in ("mean", "std", "min", "max"):
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, torch.Tensor):
                return tuple(v.shape)
            else:
                return tuple(np.asarray(v).shape)
    return None

def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
    *,
    rel_actions: bool = False,
    action_chunk: int | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args: (see Normalize and Unnormalize)

    Returns:
        dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.

    If `rel_actions=True` we allow an extra leading time dimension
    on the 'action' key stats: (T, A) instead of (A,). If provided, we honor the shape
    found in `stats[key]` even if it differs from `ft.shape`.
    """
    stats_buffers = {}

    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        assert isinstance(norm_mode, NormalizationMode)

        shape = tuple(ft.shape)

        if ft.type is FeatureType.VISUAL:
            # sanity checks
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # --- NEW: infer 'timewise' shape from stats or action_chunk ---
        # Prefer the shape *actually present* in stats if given.
        if stats and key in stats:
            sshape = _first_stat_shape(stats[key])
        else:
            sshape = None

        if sshape is not None:
            # Honor provided stats shape (handles (T,A) neatly)
            shape = sshape
        elif rel_actions and key == "action" and action_chunk is not None:
            # No stats shape given yet; allocate timewise buffers (T,A)
            shape = (action_chunk,) + tuple(ft.shape)

        # Create the buffer tensors
        buffer = {}
        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.MIN_MAX:
            min_ = torch.ones(shape, dtype=torch.float32) * torch.inf
            max_ = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min_, requires_grad=False),
                    "max": nn.Parameter(max_, requires_grad=False),
                }
            )

        # Load provided stats (np or torch) exactly like your original
        if stats and key in stats:
            d = stats[key]
            if isinstance(_first_stat_shape(d), tuple):  # stats present
                if norm_mode is NormalizationMode.MEAN_STD:
                    if isinstance(d["mean"], np.ndarray):
                        buffer["mean"].data = torch.from_numpy(d["mean"]).to(dtype=torch.float32)
                        buffer["std"].data = torch.from_numpy(d["std"]).to(dtype=torch.float32)
                    else:
                        buffer["mean"].data = d["mean"].clone().to(dtype=torch.float32)
                        buffer["std"].data = d["std"].clone().to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    if isinstance(d["min"], np.ndarray):
                        buffer["min"].data = torch.from_numpy(d["min"]).to(dtype=torch.float32)
                        buffer["max"].data = torch.from_numpy(d["max"]).to(dtype=torch.float32)
                    else:
                        buffer["min"].data = d["min"].clone().to(dtype=torch.float32)
                        buffer["max"].data = d["max"].clone().to(dtype=torch.float32)

        stats_buffers[key] = buffer
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


class Normalize(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
        *,
        rel_actions: bool = False,
        action_chunk: int | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
            rel_actions (bool, optional): Whether to allow an extra leading time dimension on the 'action'
                key stats: (T, A) instead of (A,). Defaults to False.
            action_chunk (int, optional): If `rel_actions=True`, the expected length of the time dimension
                (T). If not provided, and no stats are given, the action buffers will be created without
                time dimension. Defaults to None.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        self.rel_actions = rel_actions
        self.action_chunk = action_chunk

        stats_buffers = create_stats_buffers(
            features, norm_map, stats, rel_actions=rel_actions, action_chunk=action_chunk
        )
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                # FIXME(aliberts, rcadene): This might lead to silent fail!
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            x = batch[key]  # can be (B,A) or (B,T,A)

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                # Broadcasting handles per-step stats automatically:
                # (B,T,A) - (T,A)  or (B,A) - (A)
                batch[key] = (x - mean) / (std + 1e-8)

            elif norm_mode is NormalizationMode.MIN_MAX:
                min_ = buffer["min"]
                max_ = buffer["max"]
                assert not torch.isinf(min_).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_).any(), _no_stats_error_str("max")
                y = (x - min_) / (max_ - min_ + 1e-8)   # -> [0,1]
                batch[key] = y * 2 - 1                   # -> [-1,1]

            else:
                raise ValueError(norm_mode)
        return batch


class Unnormalize(nn.Module):
    """
    Similar to `Normalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
        *,
        rel_actions: bool = False,
        action_chunk: int | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
            rel_actions (bool, optional): Whether to allow an extra leading time dimension on the 'action'
                key stats: (T, A) instead of (A,). Defaults to False.
            action_chunk (int, optional): If `rel_actions=True`, the expected length of the time dimension
                (T). If not provided, and no stats are given, the action buffers will be created without
                time dimension. Defaults to None.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        self.rel_actions = rel_actions
        self.action_chunk = action_chunk

        stats_buffers = create_stats_buffers(
            features, norm_map, stats, rel_actions=rel_actions, action_chunk=action_chunk
        )
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            x = batch[key]  # (B,A) or (B,T,A)

            for stat_key in ("mean", "std", "min", "max"):
                if stat_key in buffer:
                    stat_val = buffer[stat_key]
                    # If stats have more dims than x and time dimension is larger
                    if stat_val.ndim == 3 and x.ndim >= 2:
                        if stat_val.shape[1] > x.shape[1]:
                            buffer[stat_key] = stat_val[:, :x.shape[1]]

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = x * std + mean

            elif norm_mode is NormalizationMode.MIN_MAX:
                min_ = buffer["min"]
                max_ = buffer["max"]
                assert not torch.isinf(min_).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_).any(), _no_stats_error_str("max")
                y = (x + 1) / 2
                batch[key] = y * (max_ - min_) + min_

            else:
                raise ValueError(norm_mode)
        return batch
