# PGCNMDA
Traditional Chinese Medicine prescription recommendation for Alzheimer’s disease based on network propagation and reinforcement learning

## Requirements
  * python==3.9.0
  * numpy==1.23.4
  * torch==1.8.1
  * pandas==1.5.2
  * networks == 2.8.8
  
## Data
  * The herbal data for Alzheimer’s disease were obtained from the TCM Systems Pharmacology Database (TCMSP).
  * There are 29 targets, 8255 ingredients, 499 herbs, 5366 target-ingredient associations and 4962 ingredient-herb associations.
  * The experience data for calculating Escore were available from the database from the Chinese patent medicine database (http://crds.release.daodikeji.com), which includes 290
approved prescriptions.

## Usage
  * To run the program: Run 'formula_generate.py'.
  * File 'Alzheimer_pagerankdl018.csv' shows the generated top 50 prescriptions.
  

## Contact
If you have any questions or comments, please feel free to email Cheng Yan(yancheng01@hnucm.edu.cn).



