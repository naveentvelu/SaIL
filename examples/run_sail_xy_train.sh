#!/bin/bash

TRAIN_FOLDER="../../motion_planning_datasets/forest/train"
TRAIN_ORACLE_FOLDER="../SaIL/oracle/saved_oracles/xy/forest/train"
VALIDATION_FOLDER="../../motion_planning_datasets/forest/validation"
VALIDATION_ORACLE_FOLDER="../SaIL/oracle/saved_oracles/xy/forest/validation"
MODEL_FOLDER="../SaIL/learners/trained_models/xy/forest"
TRAIN_FILE_START_NUM="0"
VALIDATION_FILE_START_NUM="800"
#PRETRAINED_MODEL=" "
RESULTS_FOLDER="../SaIL/results/xy/forest"
ORACLE_FILE_TYPE="json"

python sail_xy_train.py --train_folder  ${TRAIN_FOLDER} --train_oracle_folder  ${TRAIN_ORACLE_FOLDER} --validation_folder  ${VALIDATION_FOLDER} --validation_oracle_folder  ${VALIDATION_ORACLE_FOLDER} --model_folder ${MODEL_FOLDER} --results_folder ${RESULTS_FOLDER} --train_file_start_num ${TRAIN_FILE_START_NUM} --validation_file_start_num ${VALIDATION_FILE_START_NUM} --oracle_file_type ${ORACLE_FILE_TYPE}
