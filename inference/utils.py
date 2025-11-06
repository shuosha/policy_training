from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform


class InferenceTransform(Transform):
    def __init__(self, inference_cfg, policy_type, device=None):
        super().__init__()
        self.inference_cfg = inference_cfg
        self.use_augmentation = policy_type == "lerobot"
        self.device = device

        # process raw images to the dataset size
        if "fixed_pre_crop" in inference_cfg:
            assert "wrist_pre_crop" in inference_cfg
            self.fixed_pre_crop = inference_cfg["fixed_pre_crop"]
            self.wrist_pre_crop = inference_cfg["wrist_pre_crop"]
        else:
            self.fixed_pre_crop = inference_cfg["pre_crop"]
            self.wrist_pre_crop = inference_cfg["pre_crop"]
        self.pre_resize = v2.Resize(size=inference_cfg["pre_resize"])

        # crop and resize as in training
        self.crop = v2.CenterCrop(size=inference_cfg["crop"])
        self.resize = v2.Resize(size=inference_cfg["resize"])

        self.use_overlay = inference_cfg.get("overlay", None) is not None
        if self.use_overlay:
            overlay = cv2.imread(str(Path(__file__).parents[1] / inference_cfg['overlay']), cv2.IMREAD_UNCHANGED)  # BGRA
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
            overlay = np.array(overlay).astype(np.float32) / 255.0
            overlay = torch.from_numpy(overlay)
            overlay = overlay.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 4, H, W)
            self.overlay_image = overlay

    def forward(self, x, camera_type):
        assert camera_type in ["fixed", "wrist"], f"Invalid camera type: {camera_type}"
        if self.use_overlay and camera_type == "fixed":
            overlay_alpha = self.overlay_image[:, 3:4]
            overlay_rgb = self.overlay_image[:, :3]
            blended = 0.5 * overlay_rgb + 0.5 * x
            x = blended * overlay_alpha + x * (1 - overlay_alpha)

        if camera_type == "fixed":
            x = x[..., self.fixed_pre_crop[0]:self.fixed_pre_crop[1], self.fixed_pre_crop[2]:self.fixed_pre_crop[3]]
        else:  # wrist
            x = x[..., self.wrist_pre_crop[0]:self.wrist_pre_crop[1], self.wrist_pre_crop[2]:self.wrist_pre_crop[3]]
        x = self.pre_resize(x)

        if self.use_augmentation:
            x = self.crop(x)
            x = self.resize(x)
        return x
    
    def overlay(self, image):
        if self.use_overlay:
            overlay_alpha = self.overlay_image[0, 3:4]
            overlay_rgb = self.overlay_image[0, :3]
            blended = 0.5 * overlay_rgb + 0.5 * image
            image = blended * overlay_alpha + image * (1 - overlay_alpha)
        return image


def q_normalize(q):
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def q_conj(q):  # (w, x, y, z) -> (w, -x, -y, -z)
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def q_mul(q1, q2):
    # (w1,x1,y1,z1) * (w2,x2,y2,z2)
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def q_apply(q, v):
    # rotate vector v by quaternion q (wxyz)
    # v' = q * (0,v) * q_conj
    zeros = torch.zeros_like(v[..., :1])
    v_as_quat = torch.cat([zeros, v], dim=-1)
    return q_mul(q_mul(q, v_as_quat), q_conj(q))[..., 1:]
