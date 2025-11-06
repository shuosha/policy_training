# Training on DROID

Here we describe how to fine-tune the pi0-FAST model on the DROID dataset. This is an approximate open-source reproduction of the pi0-FAST-DROID training pipeline.
(small differences in data loading and the used action space)

In contrast to the rest of openpi, which uses LeRobot for data loading, we need to use RLDS as the data format for DROID training (since atm LeRobot isn't scalable enough 
for larger datasets like DROID -- they are working on improving it though). Below, we provide instructions for updating your openpi environment for RLDS data loading and where to download the DROID dataset.

## Install

We need a few additional dependencies for RLDS data loading. Run:
```bash
uv sync --group rlds
```

## Download DROID dataset

You can download the DROID dataset with the following command (after installing the `gsutil` google cloud CLI):
```
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 <your_download_path>/droid/1.0.1
```

Note that downloading version 1.0.1 is important (not v1.0.0): it contains the complete set of language annotations (~75k episodes) while v1.0.0 only has annotations for 30k episodes. If for some reason you would like to use another version, modify the line `version="1.0.1"` in the `DroidRldsDataset` object [here](src/openpi/training/droid_rlds_dataset.py).

You will need 1.8TB of disk storage to download the DROID RLDS dataset.

## Run

First, change the `rlds_data_dir` path in your `TrainConfig` to the directory that you downloaded the `droid` dataset into (see [src/openpi/training/config.py](src/openpi/training/config.py)).

Then, compute normalization statistics (this will take ~10 minutes):
```bash
uv run --group rlds scripts/compute_norm_stats.py --config-name pi0_fast_droid_finetune
```

Run training:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py pi0_fast_droid_finetune --exp-name=my_experiment --overwrite
```

**Note**: The original pi0-FAST-DROID model was trained with joint velocity actions.
Joint velocity actions are not compatible with simulated evaluation environments (much harder to simulate). 
Thus, we do not recommend training with joint velocity actions and instead use joint position actions here.


## Compute Requirements

Our DROID training config requires approximately 2 days on 8x H100 GPUs for convergence (100k iterations, bs256, approx. 1 epoch).
If you start from PaliGemma instead of pi0 initialization, plan with ~5 days on 8x H100s (240k iterations, i.e. 3 epochs).

We have experimented with LoRA for cheaper finetuning, but haven't found the policies to perform well so far.


## Data Filtering

Like any diverse real-robot dataset, the DROID dataset isn't perfectly "clean" and we have found data filtering to significantly improve policy performance. Concretely, the DROID dataset contains many *idle* timesteps in which the robot does not move (in part due to the VR teleoperation interface that was used during data collection, we will not go into too much detail here). Appropriate filtering of these idle transitions can improve policy performance.

By default, our openpi training recipe implements the same idle filter used to train all pi-DROID models. We implement it by pre-computing which dataset indices to sample during training. You can check [compute_droid_nonidle_ranges.py](examples/droid/compute_droid_nonidle_ranges.py) for how we compute these indices. Roughly speaking, we filter any time steps for which the next chunk of actions would be largely idle. During training, our code automatically pulls our pre-computed list of indices from cloud storage and applies them. If you want to modify the idle filter / create your custom sampling logic, you can modify our script to generate a new index list and provide it via the `filter_dict_path="<path_to_filter_dict>"` argument in [src/openpi/training/config.py](src/openpi/training/config.py).

**Note**: our list of filtering indices is only valid for the `droid/1.0.1` dataset mentioned in the download section above, and will not provide valid filtering for any other version of the DROID dataset, so make sure you download the dataset above! If you have a custom DROID version, you can rerun the [compute_droid_nonidle_ranges.py](examples/droid/compute_droid_nonidle_ranges.py) script to generate a new list of sampling indices.

## RoboArena

Consider submitting your DROID policies to the [RoboArena benchmark](https://robo-arena.github.io/), which allows you to evaluate your policies on diverse tasks & scenes, **in the real world**! :)

If you have questions about RoboArena, please email [karl.pertsch@gmail.com](mailto:karl.pertsch@gmail.com).
