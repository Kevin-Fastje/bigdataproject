# bigdataproject – Regional Renewable Generation Pattern Analysis (Germany)

This script performs a big data analysis of renewable energy generation patterns (PV, onshore wind, offshore wind) for German NUTS-2 regions. It includes a PCA-based dimensionality reduction and KMeans clustering, visualised on a geospatial map.

# Requirements:
- Python packages: pandas, numpy, matplotlib, scikit-learn, geopandas, networkx, functools
- Install dependencies via pip or conda

# Setup:
1. Download the input dataset from the GitHub repository of the Aachen study:
   https://github.com/FZJ-IEK3-VSA/Robust-Capacity-Expansion
2. Update file paths in the script to match your local directory