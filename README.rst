|PYPI| |Issues| |Codecov| |Downloads|

.. |PYPI| image:: https://badge.fury.io/py/sccellfie.svg
   :target: https://pypi.org/project/sccellfie/

.. |Issues| image:: https://github.com/earmingol/scCellFie/actions/workflows/tests.yml/badge.svg
   :alt: test-sccellfie

.. |Codecov| image:: https://codecov.io/gh/earmingol/scCellFie/graph/badge.svg?token=22NENAKNKI
   :target: https://codecov.io/gh/earmingol/scCellFie

.. |Downloads| image:: https://pepy.tech/badge/sccellfie/month
   :target: https://pepy.tech/project/sccellfie



Metabolic activity from single-cell and spatial transcriptomics with scCellFie
-----------------------------------------------------------------------------------------

scCellFie is a computational tool for studying metabolic tasks using Python, inspired by the original implementation of
`CellFie <https://github.com/LewisLabUCSD/CellFie>`_, another tool originally developed in MATLAB by the `Lewis Lab <https://lewislab.ucsd.edu/>`_. This version is designed to be
compatible with single-cell and spatial data analysis using Scanpy, while including a series of improvements and new analyses.

.. image:: https://github.com/earmingol/scCellFie/blob/main/scCellFie-Logo.png?raw=true
   :alt: Logo
   :width: 350
   :height: 188.31
   :align: center


Installation
------------

To install scCellFie, use pip::

    pip install sccellfie

Features
--------

- **Single cell and spatial data analysis:** Tailored for analysis of metabolic
  tasks using fully single cell resolution and in space.

- **Speed:** This implementation further leverages the original CellFie. It is now memory
  efficient and run much faster! A dataset of ~70k single cells can be analyzed in ~5 min.

- **New analyses:** From marker selection of relevant metabolic tasks to integration with
  inference of cell-cell communication.

- **User-friendly:** Python-based for easier use and integration into existing workflows.

- **Scanpy compatibility:** Fully integrated with Scanpy, the popular single cell
  analysis toolkit.

- **Organisms:** Metabolic database and analysis available for human and mouse.

How to cite
-----------

*Preprint is coming soon!*

Acknowledgments
---------------

This implementation is inspired by the original `CellFie tool <https://github.com/LewisLabUCSD/CellFie>`_ developed by
the `Lewis Lab <https://lewislab.ucsd.edu/>`_. Please consider citing their work if you find this tool useful:

- **Model-based assessment of mammalian cell metabolic functionalities using omics data**.
  *Cell Reports Methods, 2021*. https://doi.org/10.1016/j.crmeth.2021.100040

- **ImmCellFie: A user-friendly web-based platform to infer metabolic function from omics data**.
  *STAR Protocols, 2023*. https://doi.org/10.1016/j.xpro.2023.102069

- **Inferring secretory and metabolic pathway activity from omic data with secCellFie**.
  *Metabolic Engineering, 2024*. https://doi.org/10.1016/j.ymben.2023.12.006