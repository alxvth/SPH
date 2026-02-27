# Comparison setup

1. Download the data with `dataDownload.py` from `EvalSpatialHierarchy`
    - This script will populate `EvalSpatialHierarchy/data` with the Indian Pines and Salinas data
2. Run comparisons, 
    - Part 1: `computeSuperpixels.py` from `EvalSpatialHierarchy` for ERS, FH and SLIC 
    - Part 2: `run_evaluation.m` from `BarbatoHyperspectralSLIC/code` for HyperspectralSLIC
        - Rename the output files with `renameFiles.py` from `EvalSpatialHierarchy`
    - Part 3: Run SPH using the `comparison_*.json` input scripts and evaluation executable from `../evaluation`
        -  You'll need to adjust the Paths
3. Evaluate the results with `evalSuperpixels.py` from `EvalSpatialHierarchy`
