from pathlib import Path
import json
import torch
import os
from huggingface_hub import snapshot_download

from .utils import InferenceTransform
from .policy_lerobot import LerobotPolicy
from .policy_openpi import OpenPIPolicy

class PolicyInferenceWrapper:
    def __init__(
        self,
        inference_cfg_path: str,
        checkpoint_path: str,          # EITHER a local folder (e.g., ".../checkpoints/020000")
                                       # OR an HF repo_id (e.g., "shashuo0104/svla-T-block-pushing")
        hf_subdir: str | None = None,  # REQUIRED if checkpoint_path is an HF repo_id
        local_rank: int = 0,
        hf_token: str | None = None,   # for private repos
    ):
        self.device = f"cuda:{local_rank}"

        with open(inference_cfg_path, "r") as f:
            inference_cfg = json.load(f)

        # Resolve to a local checkpoint root folder
        ckpt_root = self._resolve_ckpt_root(
            checkpoint_path=checkpoint_path,
            hf_subdir=hf_subdir,
            hf_token=hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        )

        # Detect layout and load policy
        if (ckpt_root / "pretrained_model").is_dir():
            self.policy_type = "lerobot"
            self.policy = LerobotPolicy(inference_cfg, str(ckpt_root), self.device)
        elif (ckpt_root / "params").is_dir():
            self.policy_type = "openpi"
            self.policy = OpenPIPolicy(inference_cfg, str(ckpt_root), self.device)
        else:
            raise ValueError(f"Invalid checkpoint layout: {ckpt_root}")

        self.policy.to(self.device)

        self.obs_dict = {
            "observation.images.front": None,
            "observation.images.wrist": None,
            "observation.state": None,
        }
        self.tf = InferenceTransform(inference_cfg, self.policy_type, self.device)

    # --- helpers ---
    def _resolve_ckpt_root(self, checkpoint_path: str, hf_subdir: str | None, hf_token: str | None) -> Path:
        p = Path(checkpoint_path)
        if p.exists():
            # Local path given → use as-is
            return p.resolve()

        # Treat as HF repo id; require hf_subdir
        if not hf_subdir:
            raise ValueError(
                "checkpoint_path does not exist locally and looks like a repo id; "
                "please provide hf_subdir (e.g., 'checkpoints/020000')."
            )

        # Download only the subdir to keep things fast & small
        local_dir = f"./_hf_cache_{checkpoint_path.split('/')[-1]}"
        snap_dir = snapshot_download(
            repo_id=checkpoint_path,          # e.g., "shashuo0104/svla-T-block-pushing"
            repo_type="model",
            allow_patterns=[f"{hf_subdir}/**"],
            local_dir=local_dir,
            token=hf_token,
        )
        root = Path(snap_dir) / hf_subdir
        if not root.exists():
            # Fallback: grab everything if filtered download missed it
            snap_dir = snapshot_download(
                repo_id=checkpoint_path,
                repo_type="model",
                local_dir=local_dir,
                token=hf_token,
            )
            root = Path(snap_dir) / hf_subdir

        if not root.exists():
            raise FileNotFoundError(f"Downloaded but cannot find checkpoint subdir: {root}")
        return root.resolve()

    def reset(self):
        self.policy.reset()

    def inference(self, obs_dict: dict):
        """
        Input: 
            obs_dict: {
                "observation.images.front": tensor (1, 3, 480, 848), 
                "observation.images.wrist": tensor (1, 3, 480, 848),
                "observation.state": tensor (1, 8),
            }
        Output:
            action: tensor (1, 8) in absolute cartesian coordinates
        """
        assert set(obs_dict.keys()) == set(self.obs_dict.keys()), \
            f"Expected keys {self.obs_dict.keys()}, got {obs_dict.keys()}"

        obs_dict["observation.images.front"] = self.tf(
            obs_dict["observation.images.front"].to(device=self.device),
            camera_type="fixed",
        )

        obs_dict["observation.images.wrist"] = self.tf(
            obs_dict["observation.images.wrist"].to(device=self.device),
            camera_type="wrist",
        )

        cartesian_goal = self.policy.select_action(obs_dict)
        return cartesian_goal

    def visualize_overlay(self, image: torch.Tensor):
        return self.tf.overlay(image)
