## Robust PCA for Adverse Weather Image Analysis (DAWN Dataset)

### Overview

This project analyzes images from the DAWN adverse-weather dataset using
Robust PCA / Principal Component Pursuit (PCP) to decompose each image into:

Low-rank component (L) — underlying scene structure

Sparse component (S) — weather artifacts such as rain, snow, dust, or noise

This repository contains the full pipeline including preprocessing, PCP implementation, visualization tools, and quantitative metrics for comparing decomposition behavior across weather types.

### Folder Structure
down_pcp_project/
│
├── notebooks/
│   └── pcp_analysis.ipynb        # Main notebook with full analysis
│
├── src/
│   ├── io_utils.py               # Image loading & preprocessing
│   ├── pcp.py                    # PCP implementation (inexact ALM)
│   ├── metrics.py                # Metrics: rank, sparse energy, sparsity
│   ├── viz.py                    # Visualization utilities
│   └── utils.py                  # (Optional) helpers
│
├── data/                         # (Ignored by Git) Place DAWN images here
│   ├── rain/
│   ├── snow/
│   ├── fog/
│   ├── dust/
│   └── clear/
│
├── .gitignore
├── requirements.txt
└── README.md


⚠️ Important:
The data/ folder is NOT included in the repository.
You must download and place the DAWN dataset locally before running the notebook.

📥 Dataset Instructions (How to Download & Store)

Download the DAWN dataset from:
[https://github.com/visionlab-ucr/dawn](https://www.kaggle.com/datasets/shuvoalok/dawn-dataset)

Extract the dataset.

Inside this repository, create the following structure:

down_pcp_project/data/images/
    dusttornado/
    foggy/
    haze/
    mist/
    rain_storm/
    sand_storm/
    snow_storm/


Copy the images into their corresponding folders.

Your image paths should look like:

down_pcp_project/data/images/foggy/foggy-001.jpg
...


The pipeline automatically discovers all categories inside data/.
