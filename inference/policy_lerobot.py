from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .utils import q_normalize, q_mul, q_apply


class LerobotPolicy(nn.Module):

    def __init__(self,
        inference_cfg,
        checkpoint_path,
        device
    ):
        super().__init__()
        self.device = device

        # load training config
        cfg_path = Path(checkpoint_path) / "pretrained_model" / "train_config.json"
        with open(cfg_path) as f: 
            train_cfg = json.load(f)
        ptype = train_cfg["policy"]["type"]
        self.task_name = inference_cfg["task_name"]
        self.chunk_size = inference_cfg.get("chunk_size", 50)

        cls_map = {
            "diffusion": DiffusionPolicy,
            "act": ACTPolicy,
            "smolvla": SmolVLAPolicy
        }
        if ptype not in cls_map: 
            raise NotImplementedError(f"Unsupported policy type: {ptype}")

        # relative action
        rel_stats = None
        self.use_relative_action = train_cfg["policy"].get("relative_actions", False)
        if self.use_relative_action:
            stats = Path(checkpoint_path) / "assets" / "relative_action_stats.pt"
            rel_stats = torch.load(stats)

        Policy = cls_map[ptype]
        self.policy_model = Policy.from_pretrained(
            str(Path(checkpoint_path) / "pretrained_model"),
            device=self.device,
            **({"rel_action_stats": rel_stats} if rel_stats is not None else {})
        )
        inp, out = self.policy_model.config.input_features, self.policy_model.config.output_features
        self.obs_dict = {k: None for k in inp.keys()}
        print(f"Loaded {ptype} policy from {Path(checkpoint_path)}, input: {inp}, output: {out}")

        self.action_dim = inference_cfg["action_dim"]
        self.fixed_pos_z = inference_cfg["fixed_pos_z"]
        self.idx = 0

    def reset(self):
        self.idx = 0
        self.policy_model.reset()

    def select_action(self, obs_dict):
        obs_dict["task"] = [self.task_name]
        with torch.no_grad():
            action = self.policy_model.select_action(obs_dict)  # (1, 8)

        # for planer pushing
        if self.action_dim == 2:
            if self.use_relative_action:
                if self.idx % self.chunk_size == 0:
                    self._p0 = obs_dict["observation.state"][:,:2].clone()

                pos_xy = self._p0 + action[:,:2]
                pos_xy = pos_xy.reshape(-1).cpu().numpy()
            else:
                pos_xy = action.reshape(-1).cpu().numpy() #(2, )

            r, p, y = -np.pi, 0.0, 0.0
            cartesian_np = np.concatenate([pos_xy, [self.fixed_pos_z, r, p, y]], axis=0)  # (6,)
            cartesian_goal = torch.tensor(cartesian_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        else:
            if self.use_relative_action:
                if self.idx % self.chunk_size == 0:
                    self._p0 = obs_dict["observation.state"][:,:3].clone()
                    self._q0 = q_normalize(obs_dict["observation.state"][:,3:7].clone())
                    self._g0 = obs_dict["observation.state"][:,-1:].clone()

                pt = self._p0 + q_apply(self._q0, action[:, :3])
                qt = q_mul(self._q0, q_normalize(action[:, 3:7]))
                gt = action[:, -1:]
                cartesian_goal = torch.cat([pt, qt, gt], dim=-1)
            else:
                cartesian_goal = action  # (1, 8)

        self.idx += 1
        return cartesian_goal
