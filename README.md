# IL Policy Training Framework

This repository provides a **unified environment** for training and evaluating **visuomotor policies** — such as **Diffusion Policy**, **ACT**, **SmolVLA**, and **Pi0** — under a common interface.

---

## 1. Python Environment Setup

### Clone the Repository

```bash
git clone git@github.com:shuosha/policy_training.git
cd policy_training
````

### Install `uv`

Follow the official [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Create and Sync the Environment

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

> **Note:** Ensure your `torchcodec` version is compatible with your installed PyTorch and Python versions.
> See [meta-pytorch/torchcodec](https://github.com/meta-pytorch/torchcodec) for version compatibility details.

---

## 2. Experiment Tracking (Optional)

To enable experiment logging with **Weights & Biases (wandb)**:

```bash
wandb login
```

Once logged in, all training runs will automatically sync to your WandB account.

---

## 3. Training Your Own Models

Training datasets for all tasks are hosted on Hugging Face and are automatically downloaded during training.

| Task                | Hugging Face Dataset Collection                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| **Rope Routing**    | [shashuo0104/xarm7-insert-rope](https://huggingface.co/datasets/shashuo0104/xarm7_insert_rope)       |
| **Toy Packing**     | [shashuo0104/xarm7-pack-sloth](https://huggingface.co/datasets/shashuo0104/xarm7_pack_sloth)         |
| **T-Block Pushing** | [shashuo0104/xarm7-pusht](https://huggingface.co/datasets/shashuo0104/xarm7_pusht) |

> Datasets are automatically downloaded through the training scripts; manual download is not required.

---

### 3.1 Launch Training

All policies share a unified command-line interface for training:

```bash
bash scripts/train_<policy_name>.sh <task_name> <experiment_name>
```

**Arguments:**

* `<policy_name>` ∈ {`dp`, `act`, `svla`, `pi0`}
  → Policy type (Diffusion Policy, Actiong Chunking Transformer, SmolVLA, or Pi0).
* `<task_name>` ∈ {`insert_rope`, `pack_sloth`, `pusht`}
  → Specifies which dataset/environment to train on.
* `<experiment_name>`
  → Custom label for logs and checkpoints.

**Example:**

```bash
bash scripts/train_act.sh insert_rope demo_run
```

This launches **ACT** training on the **Rope Routing** dataset and saves checkpoints under:

```
outputs/checkpoints/insert_rope/<timestamp>_act_demo_run/
```

**Video Tools:**
Ensure **FFmpeg** is installed on your system or environment — it is required for dataset video preprocessing, episode collation, and rollout visualization.
You can verify installation with:

```bash
ffmpeg -version
```

and install it (if missing) via:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg -y
```

---


### 3.2 Configuration Files

Configuration files for each task and policy are located under:

```
configs/training/<policy_name>_<task_name>.cfg
```

| Variable        | Description                                           |
| --------------- | ----------------------------------------------------- |
| `<task_name>`   | One of `{insert_rope, pack_sloth, pusht}` |
| `<policy_name>` | One of `{dp, act, svla, pi0}`                         |

Each configuration defines:

* Model architecture and hyperparameters
* Dataset loader and preprocessing settings
* Training schedule, batch size, and learning rate

> **Hardware Note:** Adjust `num_workers` and `batch_size` to match your GPU capacity.
> Default configs have been validated on an **NVIDIA RTX 5090**.

---

## 4. Policy Inference

Once trained, policies can be evaluated in two ways:
* Checkpoints you trained locally, or
* Pretrained checkpoints downloaded from [Hugging Face](https://huggingface.co/collections/shashuo0104/real-to-sim-policy-eval).

### 4.1 Using Local or Pretrained Checkpoints

| Policy                                | Rope Routing                                                              | Toy Packing                                                             | T-Block Pushing                                                                 |
| :------------------------------------ | :------------------------------------------------------------------------ | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| **Diffusion Policy**             | [dp-insert-rope](https://huggingface.co/shashuo0104/dp-insert-rope)     | [dp-pack-sloth](https://huggingface.co/shashuo0104/dp-pack-sloth)     | [dp-pusht](https://huggingface.co/shashuo0104/dp-pusht)     |
| **Action Chunking Transformer** | [act-insert-rope](https://huggingface.co/shashuo0104/act-insert-rope)   | [act-pack-sloth](https://huggingface.co/shashuo0104/act-pack-sloth)   | [act-pusht](https://huggingface.co/shashuo0104/act-pusht)   |
| **SmolVLA**                    | [svla-insert-rope](https://huggingface.co/shashuo0104/svla-insert-rope) | [svla-pack-sloth](https://huggingface.co/shashuo0104/svla-pack-sloth) | [svla-pusht](https://huggingface.co/shashuo0104/svla-pusht) |
| **Pi0**                      | [pi0-insert-rope](https://huggingface.co/shashuo0104/pi0-insert-rope)   | [pi0-pack-sloth](https://huggingface.co/shashuo0104/pi0-pack-sloth)   | [pi0-pusht](https://huggingface.co/shashuo0104/pi0-pusht)   |

#### Example (local or downloaded checkpoints)

```python
from inference.inference_wrapper import PolicyInferenceWrapper

policy = PolicyInferenceWrapper(
    inference_cfg_path="configs/inference/insert_rope.json",
    checkpoint_path="outputs/checkpoints/<timestamp>-act-insert-rope/010000/"  # or downloaded HF dir
)
```

> **Note:** `checkpoint_path` should point to the checkpoint folder (e.g., `010000/`).

---

#### 💡 How to Download from Hugging Face

**Option 1: Using `git lfs`**

```bash
sudo apt install git-lfs
git lfs install
mkdir outputs && mkdir outputs/checkpoints && cd outputs/checkpoints
git clone https://huggingface.co/shashuo0104/svla-pusht
```

**Option 2: Using Python API**

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="shashuo0104/svla-pusht",
    repo_type="model",
    local_dir="outputs/checkpoints"
)
```

---

### 4.2 Using Checkpoints Directly from Hugging Face

You can load checkpoints directly from Hugging Face without manual download.
The wrapper automatically fetches the checkpoint subdirectory using `huggingface_hub`.

```python
from inference.inference_wrapper import PolicyInferenceWrapper

policy = PolicyInferenceWrapper(
    inference_cfg_path="configs/inference/pusht.json",
    checkpoint_path="shashuo0104/pi0-pusht",  # HF repo ID
    hf_subdir="20000"                         # points to a specific checkpoint folder
)
```

---

### Run Inference

```python
cartesian_action = policy.inference(obs_dict)
```

#### Expected Input Format

```python
obs_dict = {
    "observation.images.front": tensor(1, 3, 480, 848),
    "observation.images.wrist": tensor(1, 3, 480, 848),
    "observation.state": tensor(1, action_dim),
}
```

#### Output

`cartesian_action`: a tensor of shape `(1, action_dim)` where:

| Task Type                      | `action_dim` | Description                                          |
| ------------------------------ | ------------ | ---------------------------------------------------- |
| **Rope Routing / Toy Packing** | 8            | `[eef_pos (3), eef_quat (4, wxyz), gripper_pos (1)]` |
| **T-Block Pushing**            | 2            | `[eef_xy]` (z = 0.22 m, quat = [1, 0, 0, 0])         |

---

### Special Note for Push-T

* The raw Push-T images **do not include the “T” goal marker**.
* During training and inference, the goal image is **overlaid** on the front camera view.
* The reference goal image is located at:

  ```
  pusht_masks/pushT_goal.png
  ```

---