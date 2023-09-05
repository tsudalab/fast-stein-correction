# !/bin/bash

for PROBLEM in GSD_8 GSD_38 GSD_F_6;
do
  mkdir ${PROBLEM}_outputs
  papermill mcmc.ipynb ${PROBLEM}_outputs/mcmc.ipynb -r PROBLEM ${PROBLEM}
done;
