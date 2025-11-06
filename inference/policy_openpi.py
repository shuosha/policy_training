from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from openpi.training import config as pi0_cfg
from openpi.policies import policy_config
from openpi.shared import download


class OpenPIPolicy(nn.Module):

    def __init__(self,
        inference_cfg,
        checkpoint_path,
        device
    ):
        super().__init__()
        self.device = device

        self.task_name = inference_cfg["task_name"]

        self.obs_dict = {
            "observation.images.front": None,
            "observation.images.wrist": None,
            "observation.state": None,
        }

        config = pi0_cfg.get_config(f"pi0_lora_{self.task_name}")
        checkpoint_dir = download.maybe_download(checkpoint_path)
        self.policy_model = policy_config.create_trained_policy(config, checkpoint_dir)

        self.action_dim = inference_cfg["action_dim"]
        self.fixed_pos_z = inference_cfg["fixed_pos_z"]
        self.action_queue = []
        self.idx = 0

    def reset(self):
        self.idx = 0
        self.policy_model.reset()

    def select_action(self, obs_dict):
        # manually pass through action queue for pi0
        for k, v in obs_dict.items():
            v_transformed = v.detach().clone()[0]
            if k != "observation.state":
                v_transformed = v_transformed.permute(1, 2, 0)
            v_transformed = v_transformed.cpu().numpy()
            obs_dict[k] = v_transformed
        obs_dict["task"] = self.task_name

        if len(self.action_queue) == 0:
            with torch.no_grad():
                policy_output = self.policy_model.infer(obs_dict)
            action_chunk = policy_output["action"]
            action_chunk = torch.from_numpy(action_chunk).float().to(self.device)
            self.action_queue = [action_chunk[i] for i in range(action_chunk.shape[0])]

        action = self.action_queue.pop(0).unsqueeze(0)  # (1, action_dim)

        # for planer pushing
        if self.action_dim == 2:
            pos_xy = action.reshape(-1).cpu().numpy() #(2, )
            r, p, y = -np.pi, 0.0, 0.0
            cartesian_np = np.concatenate([pos_xy, [self.fixed_pos_z, r, p, y]], axis=0)  # (6,)
            cartesian_goal = torch.tensor(cartesian_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            cartesian_goal = action  # (1, 8)

        self.idx += 1
        return cartesian_goal
