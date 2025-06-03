#!/bin/bash
set -x -e


ROOT=/project/hippogang_2/mdong/Longi_T1_Aim2/attention_heatmap
MODEL_DATE="2022-04-08_02-11"
CHECK_POINT="0010"
DATADIR=${ROOT}/data/${MODEL_DATE}_val_${CHECK_POINT}   # model_name, which epoch of evaluation

param1=3mm
param2=0.05mm

OUTDIR=${ROOT}/out/${param1}_${param2}
ATTN_DIR=${ROOT}/out/${MODEL_DATE}_val_${CHECK_POINT}_reg_${param1}_${param2}
ATTN_MAPs=$(ls -d -1 ${ATTN_DIR}/*to_hw_reslice_attention.nii.gz)


for ATTN_MAP in ${ATTN_MAPs}
do

  FILE_NAME=$(echo ${ATTN_MAP} | cut -d'/' -f 9 | cut -d'.' -f 1)
  echo $FILE_NAME

  # in attention.nii.gz, separate hippo region and sulcus region
  c3d ${ATTN_MAP} -split -oo ${ATTN_DIR}/${FILE_NAME}_%03d.nii.gz

done
  
# build clouds for both regions and both sides
HIPPO_MAP_LEFT=$(ls -d -1 ${ATTN_DIR}/*_left_to_hw_reslice_attention_001.nii.gz)
HIPPO_MAP_RIGHT=$(ls -d -1 ${ATTN_DIR}/*_right_to_hw_reslice_attention_001.nii.gz)

SULCUS_MAP_LEFT=$(ls -d -1 ${ATTN_DIR}/*_left_to_hw_reslice_attention_002.nii.gz)
SULCUS_MAP_RIGHT=$(ls -d -1 ${ATTN_DIR}/*_right_to_hw_reslice_attention_002.nii.gz)


c3d ${HIPPO_MAP_LEFT} -accum -add -endaccum -o ${ATTN_DIR}/hippo_map_left.nii.gz
c3d ${HIPPO_MAP_RIGHT} -accum -add -endaccum -o ${ATTN_DIR}/hippo_map_right.nii.gz

c3d ${SULCUS_MAP_LEFT} -accum -add -endaccum -o ${ATTN_DIR}/sulcus_map_left.nii.gz
c3d ${SULCUS_MAP_RIGHT} -accum -add -endaccum -o ${ATTN_DIR}/sulcus_map_right.nii.gz










