# Metagenomics Classification Benchmark EDA

## Introduction

The present repository was created as an accompaniment to the paper "INSaFLU-TELEVIR: an open web-based bioinformatics suite for viral metagenomic detection and routine genomic surveillance", where the TELEVIR module is introduced. INSaFLU-TELEVIR is a web-based bioinformatics suite for viral metagenomic detection and routine genomic surveillance ([website](https://insaflu.insa.pt/); [github](https://github.com/INSaFLU/); [documentation](https://insaflu.readthedocs.io/en/latest)). The present repository contains the code used to perform the exploratory data analysis (EDA) of the metagenomics classification benchmark.

## Data

For this benchmark we relied on human and animal samples spanning a variety of pathogens and hosts. A total of 20 nanopore samples and 24 illumina samples were tested.

## Code

The code used to perform the EDA is available in the `notebooks` folder. The code is divided into multiple notebooks exploring different aspects of the benchmark. The technology explored in each is determined in the second cell of each notebook- `tech = 'nanopore'` or `tech = 'illumina'`.
