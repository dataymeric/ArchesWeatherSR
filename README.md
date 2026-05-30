# Super-Resolving Coarse-Resolution Weather Forecasts with Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-2604.00897-b31b1b.svg)](https://arxiv.org/abs/2604.00897)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-dataymeric%2FArchesWeatherSR-ffd21e)](https://huggingface.co/dataymeric/ArchesWeatherSR)

ArchesWeatherSR is a flow matching–based generative super-resolution model for global weather forecasts. It takes a coarse-resolution (1.5°) forecast and generates an ensemble of plausible high-resolution (0.25°) fields, recovering fine-scale variability while preserving large-scale structure. As demonstrated on [ArchesWeatherGen](https://github.com/INRIA/geoarches) forecasts in the companion paper.

![](assets/archesweathersr_overview.png)

For more information, see the [geoarches](https://geoarches.readthedocs.io/en/latest/) repository and documentation.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) to manage the environment. After cloning the repo, run:

```bash
git clone https://github.com/dataymeric/ArchesWeatherSR.git
cd archesweathersr
uv sync
```

`uv sync` installs all dependencies and the `archesweathersr` package itself in editable mode, which is required so that imports resolve correctly when running the training and inference scripts.

## Training

Configuration is managed as in [geoarches](https://geoarches.readthedocs.io/en/latest/user_guide/#hydra), using [Hydra](https://hydra.cc). The entry point is `train.py`.

### Basic run

```bash
python train.py \
  module=archesweathersr \
  dataloader=era5downscaling-hdf5 \
  ++name=my_run
```

### Data

ERA5 data was obtained from [WeatherBench2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html). We recommend downloading the data in HDF5 format for use with the `dataloaders.era5_hdf5` dataloader. We provide a small download script with `scripts/dl_era.py`.

## Inference

### Pretrained model

A pretrained model is available on [Hugging Face](https://huggingface.co/dataymeric/ArchesWeatherSR). Download the weights with:

```bash
hf download dataymeric/ArchesWeatherSR --local-dir runs/archesweathersr
```

Then run inference pointing to that directory:

```bash
python train.py mode=test ++name=archesweathersr
```

### Super-resolving ArchesWeatherGen forecasts

`archesweathersr.inference.infer_forecasts` super-resolves forecasts produced by ArchesWeatherGen. Each run processes one time slice (identified by `--task-id`) across all input files:

```bash
python -m archesweathersr.inference.infer_forecasts --task-id 0
```

To process multiple time slices in parallel (e.g. as a SLURM job array), pass `$SLURM_ARRAY_TASK_ID` as the task ID:

```bash
# run_sr.sbatch
#!/bin/bash
#SBATCH --array=0-599

python -m archesweathersr.inference.infer_forecasts --task-id $SLURM_ARRAY_TASK_ID
```

We provide a script to produce ArchesWeatherGen rollouts in the correct format for super-resolution with `scripts/rollout_archesweathergen.py`. This requires [downloading the pretrained models](https://geoarches.readthedocs.io/en/latest/archesweather/setup/#2-download-pretrained-models).

## Citation

```bibtex
@preprint{delefosse2026archesweathersr,
  title         = {Super-Resolving Coarse-Resolution Weather Forecasts With Flow Matching},
  author        = {Delefosse, Aymeric and Charantonis, Anastase and B{\'e}r{\'e}ziat, Dominique},
  year          = {2026},
  eprint        = {2604.00897},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  doi           = {10.48550/arXiv.2604.00897}
}
```