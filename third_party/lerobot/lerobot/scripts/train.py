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
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import json

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

import torch.utils.data._utils.pin_memory as __pm
__pm.pin_memory = lambda data, device=None: data

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.io\."
)

# ---------------- Quaternion helpers (wxyz) ----------------
def _q_normalize(q, eps=1e-12):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

def _q_conj(q):
    # q = [w, x, y, z]
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def _q_mul(q1, q2):
    # Hamilton product, wxyz
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def _quat_apply(q, v):
    """Rotate vector v by unit quaternion q (wxyz). Shapes broadcast on last dim."""
    w = q[..., :1]          # (..,1)
    xyz = q[..., 1:]        # (..,3)
    uv  = torch.cross(xyz, v, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)

def abs_to_rel_actions(batch: dict, pusht=False) -> dict:
    """
    Convert absolute actions to relative actions w.r.t. obs_0 for each batch item.

    Expects:
      batch['observation.state']: (B, num_obs, 8)  -> [pos(3), quat_wxyz(4), gripper(1)]
      batch['action']            : (B, T, 8)       -> [pos(3), quat_wxyz(4), gripper(1)]

    Returns:
      new_batch (dict) with batch['action'] replaced by relative actions of same shape.
    """
    assert 'observation.state' in batch and 'action' in batch, "Missing required keys."
    state  = batch['observation.state']
    action = batch['action']

    if pusht:
        assert (state.ndim  == 3 or state.ndim == 2) and state.size(-1)  == 2, f"state must be (B, num_obs, 2), got {tuple(state.shape)}"
        assert action.ndim == 3 and action.size(-1) == 2, f"action must be (B, T, 2), got {tuple(action.shape)}"
        if state.ndim == 2:
            B, _ = state.shape
            num_obs = 1
        else:
            B, num_obs, _ = state.shape
        B2, T, _      = action.shape
        assert B == B2, "Batch size mismatch between state and action."

        device = action.device
        dtype  = action.dtype

        # ---------------- Split obs_0 and actions ----------------
        if state.ndim == 2:
            obs0 = state
        else:
            obs0 = state[:, 0, :]                  # (B, 8)
        p0   = obs0[:, :2]                     # (B, 2)
        pt  = action[..., :2]                 # (B, T, 2)
        rel_action = pt - p0.unsqueeze(1)             # (B, T, 2)

        new_batch = dict(batch)
        new_batch['action'] = rel_action
        
        return new_batch

    else:
        assert (state.ndim  == 3 or state.ndim == 2) and state.size(-1)  == 8, f"state must be (B, num_obs, 8), got {tuple(state.shape)}"
        assert action.ndim == 3 and action.size(-1) == 8, f"action must be (B, T, 8), got {tuple(action.shape)}"
        if state.ndim == 2:
            B, _ = state.shape
            num_obs = 1
        else:
            B, num_obs, _ = state.shape
        B2, T, _      = action.shape
        assert B == B2, "Batch size mismatch between state and action."
        assert num_obs == 1, f"num_obs must be 1 (relative only to obs_0), got {num_obs}"

        device = action.device
        dtype  = action.dtype

        # ---------------- Split obs_0 and actions ----------------
        if state.ndim == 2:
            obs0 = state
        else:
            obs0 = state[:, 0, :]                  # (B, 8)
        p0   = obs0[:, :3]                     # (B, 3)
        q0   = obs0[:, 3:7]                    # (B, 4) wxyz
        g0   = obs0[:, 7]                      # (B,)

        pt   = action[..., :3]                 # (B, T, 3)
        qt   = action[..., 3:7]                # (B, T, 4) wxyz
        gt   = action[..., 7]                  # (B, T)

        # Normalize quaternions
        q0   = _q_normalize(q0).to(dtype=dtype, device=device)              # (B,4)
        qt   = _q_normalize(qt).to(dtype=dtype, device=device)              # (B,T,4)
        q0c  = _q_conj(q0).unsqueeze(1)                                     # (B,1,4)

        # ---------------- Relative orientation: q_rel = q0^{-1} * qt ----------------
        q_rel = _q_mul(q0c.expand(B, T, 4), qt)                              # (B,T,4)
        q_rel = _q_normalize(q_rel)

        # ---------------- Relative position: p_rel = R(q0)^{-1} * (pt - p0) ----------------
        dp    = pt - p0.unsqueeze(1)                                        # (B,T,3)
        p_rel = _quat_apply(q0c.expand(B, T, 4), dp)                        # (B,T,3)

        # ---------------- Relative gripper: additive ----------------
        g_rel = gt

        # ---------------- Pack back ----------------
        rel_action = torch.cat([p_rel, q_rel, g_rel.unsqueeze(-1)], dim=-1) # (B,T,8)

        new_batch = dict(batch)
        new_batch['action'] = rel_action
        return new_batch

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    if cfg.policy.type == "smolvla":
        # NOTE: hardcoded for now
        cfg.policy.relative_actions = True
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    # dataset.video_backend = "pyav"

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")

    use_rel_actions = cfg.policy.relative_actions
    logging.info(f"Using relative actions: {use_rel_actions}")

    if use_rel_actions:
        if cfg.policy.type == "diffusion":
            chunk_size = cfg.policy.horizon
        elif cfg.policy.type == "act" or cfg.policy.type == "pi0" or cfg.policy.type == "smolvla":
            chunk_size = cfg.policy.chunk_size

        path = dataset.root / "meta" / f"relative_action_stats_Te{chunk_size}.pt"
        rel_action_stats = torch.load(path) 

        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rel_action_stats=rel_action_stats,
        )
    else:
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
        )

    repo_id = dataset.repo_id
    if "pusht" in repo_id:
        _is_pusht = True
    else:
        _is_pusht = False

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        if use_rel_actions:
            batch = abs_to_rel_actions(batch, pusht=_is_pusht)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        # print(f"Batch {step} dataloading time: {time.perf_counter() - start_time:.3f}s")

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
