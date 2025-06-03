#!/bin/bash
set -x -e


ROOT=/project/hippogang_4/mdong/Longi_T1_Aim2/attention_heatmap
# MODEL_DATE="2022-04-29_17-14" # 0040
# MODEL_DATE="2022-04-08_02-11" # 0010, 0020, 0050
MODEL_DATE="2023-12-06_13-28" # 2022-07-27_21-17

CHECK_POINT="0024"
DATADIR=${ROOT}/data/${MODEL_DATE}_val_${CHECK_POINT}   # model_name, which epoch of evaluation

STAGE_FILE=${ROOT}/reference/Query_8labels_subject_level.csv

param1=3mm
param2=0.05mm

OUTDIR=${ROOT}/out/${MODEL_DATE}_${param1}_${param2}_test
ATTN_DIR=${OUTDIR}/${MODEL_DATE}_test_${CHECK_POINT}_reg_${param1}_${param2}_attn
ATTN_MAPs=$(ls -d -1 ${ATTN_DIR}/*to_hw_reslice_attention.nii.gz)

mkdir -p ${ATTN_DIR}/0 ${ATTN_DIR}/1 ${ATTN_DIR}/3 ${ATTN_DIR}/5


for ATTN_MAP in ${ATTN_MAPs}
do

  FILE_NAME=$(echo ${ATTN_MAP} | cut -d'/' -f 10 | cut -d'.' -f 1)
  echo $FILE_NAME
  
  subjectID=$(echo ${FILE_NAME} | cut -c 1-10)
  echo $subjectID

  # find the corresponding stage for each subject
  STAGE_DIR=$(awk -v x=${subjectID} -F, '{if ($2 == x) { print $11}  }' ${STAGE_FILE} )
  echo $STAGE_DIR  

  if [ ! -z "$STAGE_DIR" ]; then
    # in attention.nii.gz, separate hippo region and sulcus region
    c3d ${ATTN_MAP} -split -oo ${ATTN_DIR}/${STAGE_DIR}/${FILE_NAME}_%03d.nii.gz
    cp $(echo ${ATTN_MAP} | sed 's/\(.*\)attention/\1position_mask/') ${ATTN_DIR}/${STAGE_DIR}/

  fi

done

#for stage in "0" "1" "3" "5" ; do
for CURR_DIR in ${ATTN_DIR}/*/ ; do
  echo $CURR_DIR
  # build clouds for both regions and both sides
  HIPPO_MAP_LEFT=$(ls -d -1 ${CURR_DIR}*_left_to_hw_reslice_attention_001.nii.gz)
  HIPPO_MAP_RIGHT=$(ls -d -1 ${CURR_DIR}*_right_to_hw_reslice_attention_001.nii.gz)

  SULCUS_MAP_LEFT=$(ls -d -1 ${CURR_DIR}*_left_to_hw_reslice_attention_002.nii.gz)
  SULCUS_MAP_RIGHT=$(ls -d -1 ${CURR_DIR}*_right_to_hw_reslice_attention_002.nii.gz)

  POSITION_MASK_LEFT=$(ls -d -1 ${CURR_DIR}*_left_to_hw_reslice_position_mask.nii.gz)
  POSITION_MASK_RIGHT=$(ls -d -1 ${CURR_DIR}*_right_to_hw_reslice_position_mask.nii.gz)


  c3d ${HIPPO_MAP_LEFT} -accum -add -endaccum -o ${CURR_DIR}hippo_map_left.nii.gz
  c3d ${HIPPO_MAP_RIGHT} -accum -add -endaccum -o ${CURR_DIR}hippo_map_right.nii.gz

  c3d ${SULCUS_MAP_LEFT} -accum -add -endaccum -o ${CURR_DIR}sulcus_map_left.nii.gz
  c3d ${SULCUS_MAP_RIGHT} -accum -add -endaccum -o ${CURR_DIR}sulcus_map_right.nii.gz

  c3d ${POSITION_MASK_LEFT} -accum -add -endaccum -o ${CURR_DIR}position_mask_left.nii.gz
  c3d ${POSITION_MASK_RIGHT} -accum -add -endaccum -o ${CURR_DIR}position_mask_right.nii.gz

  # weighted

  c3d ${CURR_DIR}position_mask_left.nii.gz ${CURR_DIR}hippo_map_left.nii.gz -divide -o ${CURR_DIR}weighted_hippo_map_left.nii.gz
  c3d ${CURR_DIR}position_mask_right.nii.gz ${CURR_DIR}hippo_map_right.nii.gz -divide -o ${CURR_DIR}weighted_hippo_map_right.nii.gz

  c3d ${CURR_DIR}position_mask_left.nii.gz ${CURR_DIR}sulcus_map_left.nii.gz -divide -o ${CURR_DIR}weighted_sulcus_map_left.nii.gz
  c3d ${CURR_DIR}position_mask_right.nii.gz ${CURR_DIR}sulcus_map_right.nii.gz -divide -o ${CURR_DIR}weighted_sulcus_map_right.nii.gz

done







