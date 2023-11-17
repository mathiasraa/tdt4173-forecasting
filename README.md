# TDT4137 Machine Learning – Solar Energy Production Forecasting

TDT4173 Machine Learning competition @ NTNU. Solar energy production forecasting on dataset provided by ANEO.

 <a href="https://www.kaggle.com/competitions/solar-energy-production-forecasting/overview">View the competition on Kaggle</a>

## Setup

Prerequisites: Anaconda

### 1. Dataset

Download the dataset from Blackboard and drop the contents of the zip-file into the `/data` folder.

### 2. Environment

Install Mamba for package management:

```bash
conda install -n base conda-forge::mamba
```

Create the environment:

```bash
mamba env create --file env.yml
```

Activate the environment by running:

```bash
conda activate forecasting
```

In order to update this environment run:

```bash
mamba env update --file env.yml --prune
```
