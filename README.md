![test-sccellfie](https://github.com/earmingol/scCellFie/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/earmingol/scCellFie/graph/badge.svg?token=22NENAKNKI)](https://codecov.io/gh/earmingol/scCellFie)

# Metabolic functionalities of mammalian cells from single-cell and spatial transcriptomics

## About scCellFie
Single-cell CellFie is a Python implementation of [CellFie](https://github.com/LewisLabUCSD/CellFie), a tool for studying metabolic tasks 
originally developed in MATLAB by the [Lewis Lab](https://lewislab.ucsd.edu/). This version is designed to be 
compatible with single-cell and spatial data analysis using Scanpy.


<img src="./scCellFie-Logo.png" width="350" height="350" alt="Logo" style="margin-right: 10px;">


## Installation
To install scCellFie, use pip:

`pip install sccellfie`

## Features
- **Single cell and spatial data analysis:** Tailored for analysis of metabolic
tasks using fully single cell resolution and in space.

- **Scanpy compatibility:** Fully integrated with Scanpy, the popular single cell
analysis toolkit.

- **User-friendly:** Python-based for easier use and integration into existing workflows.

- **Speed:** This implementation further leverages the original CellFie. It is now memory
efficient and run much faster! A dataset of ~70k single cells can be analyzed in ~30 min.

- **New analyses:** From marker selection of relevant metabolic tasks to integration with
inference of cell-cell communication.

## Acknowledgments
This implementation is inspired by the original [CellFie tool](https://github.com/LewisLabUCSD/CellFie) developed by 
the [Lewis Lab](https://lewislab.ucsd.edu/). Please consider citing their work if you find this tool useful:

- **Model-based assessment of mammalian cell metabolic functionalities using omics data**.
*Cell Reports Methods, 2021*. https://doi.org/10.1016/j.crmeth.2021.100040
- **ImmCellFie: A user-friendly web-based platform to infer metabolic function from omics data**.
*STAR Protocols, 2023*. https://doi.org/10.1016/j.xpro.2023.102069
- **Inferring secretory and metabolic pathway activity from omic data with secCellFie**. 
*bioRxiv, 2023*. https://doi.org/10.1101/2023.05.04.539316