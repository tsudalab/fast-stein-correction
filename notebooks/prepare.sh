# !/bin/bash

for PROBLEM in GSD_8 GSD_38 GSD_F_6;
do
  mkdir ${PROBLEM}_outputs
  papermill register_exact_data.ipynb ${PROBLEM}_outputs/register.ipynb -r PROBLEM ${PROBLEM}
done;