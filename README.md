# bigdataproject – Regional Renewable Generation Pattern Analysis (Germany)

This repository contains two complementary Jupyter notebooks that together form a full exploratory–modelling workflow for German NUTS‑2 regions

analysis.ipynb: Exploratory analysis of hourly renewable generation (PV, on‑ & offshore wind). Performs PCA for dimensionality reduction, K‑Means clustering and renders the clusters on a geospatial map.

modelling.ipynb: Creates optimisation targets with Gurobi, constructs a regional graph and trains several Graph Neural Networks (GCN, GraphSAGE, GAT) with extensive hyper‑parameter search to emulate the optimiser and forecast regional generation mixes

# Requirements:
- Python core packages: pandas, numpy, matplotlib, scikit-learn, geopandas, networkx, functools, tqdm
- Deepl Learning packages: torch, torch_geometric, torch_sparse, torch_scatter, torch_cluster, torch_spline_conv
- Optimisation package: gurobipy (note: gurobipy requires a local Gurobi Optimizer licence)

- Install dependencies via pip or conda

# Setup:
1. Download the input dataset from the GitHub repository of the Aachen study:
   https://github.com/FZJ-IEK3-VSA/Robust-Capacity-Expansion
2. Update file paths in the scripts to match your local directory