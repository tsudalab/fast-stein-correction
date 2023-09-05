# !/bin/sh

ANNEALING_TIME=5
for PROBLEM in GSD_8 GSD_38 GSD_F_6;
  do
    for BETA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
    do
      INPUT_NB="experiments_qa_stein_correction.ipynb"
      OUTPUT_NB="${PROBLEM}_outputs/qa_stein_correction_${BETA}_${ANNEALING_TIME}.ipynb"
      papermill ${INPUT_NB} ${OUTPUT_NB} -r target_temperature ${BETA} -r annealing_time ${ANNEALING_TIME} -r PROBLEM ${PROBLEM}
    done;
done;
