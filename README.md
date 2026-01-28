# LRRK2-simVS-data
Data and analysis scripts for the paper "A Multistage Virtual Screening Strategy Integrating Molecular Similarity, Deep Learning Scoring, and Molecular Docking towards the Discovery of Novel LRRK2 Inhibitors".

## Repository overview
- `LRRK2_inhibitor_property_analyse/`: scripts to compute molecular properties and visualize activity-stratified distributions.
- `MD/`: inputs for molecular dynamics simulations (topology/inputs for different complexes).

## Requirements
This project depends on Python scientific packages and RDKit. Recommended installation (conda):

```bash
conda create -n lrrk2-env python=3.10 -y
conda activate lrrk2-env
conda install -c conda-forge rdkit pandas numpy seaborn matplotlib -y
```

## Usage

### 1. Compute properties and generate plots

```bash
cd LRRK2_inhibitor_property_analyse
python analyse.py
```

This reads `LRRK2_inhibitor_bindingdb.csv`, computes molecular properties and pIC50, classifies activities (`High`, `Medium`, `Low`), and writes PNG plots to the `plots/` directory.

### 2. AI-based similarity screening (Ouroboros)

```bash
export ouroboros_model="Ouroboros_M1c"
export job_name="LRRK2_query"
export profile_set="LRRK2_profile.csv" # SMILES (same to compound library) and Label/Weight
export label_col="Label" # weights for profiles in profile.csv
export compound_library="/data/home/compound_library.csv" # query library
export smiles_column="SMILES" # Specify the column name in the compound_library
export probe_cluster="Yes"
export flooding=0.5 # Only molecules exceeding this value will be considered

python -u ${ouroboros_app}/PharmProfiler.py "${ouroboros_lib}/${ouroboros_model}" "${job_name}" "${smiles_column}" "${compound_library}" "${profile_set}" "${label_col}" "${probe_cluster}" "${flooding}" 
```

The repository contains an example command for running Ouroboros profiling. See the Ouroboros project for full instructions and environment setup:

[Ouroboros on GitHub](https://github.com/Wang-Lin-boop/Ouroboros)

### 3. Molecular dynamics simulations