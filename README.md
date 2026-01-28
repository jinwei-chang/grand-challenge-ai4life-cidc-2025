# Grand-Challenge AI4Life Calcium Imaging Denoising Challenge 2025

# Installation

## Install uv 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## Install dependencies

```bash
uv sync
```

## Download Dataset

```bash
uv run ./download_data.sh
```

# Usage

## Jupyter Notebook

Open `notebook.ipynb` to execute block by block. MUST check the parameters block to setup constant variables.

## Python

**Not ready**

```bash
uv run main.py
```