#!/bin/bash
set -x -e

# This script tries to find the correct parameter setting for a pairwise registration 
# from a bl or fu image to a template image

# In the test stage, images will only be normalized and cropped, but not flipped
# left images are registered to the left template; 
# right images are registered to the right one.


SCRIPT_DIRECTORY=$(dirname "$0")

ROOT=/project/hippogang_4/mdong/Longi_T1_Aim2/attention_heatmap
MODEL_DATE="2023-12-06_13-28" # 2022-07-27_21-17, 0010, 0020, 0050
# MODEL_DATE="2022-04-29_17-14" # 0040
CHECK_POINT="0024"
DATADIR=${ROOT}/data/${MODEL_DATE}_val_${CHECK_POINT}   # model_name, which epoch of evaluation

GP_DIM=3

GP_OPT_RIGID="-n 100x50x40 -m NCC 4x4x4"
GP_OPT_AFFINE="-n 100x50x40 -m NCC 4x4x4"
param1=3mm
param2=0.05mm
GP_OPT_DEFORM=$(echo  "-n 100x50x40 -m NCC 4x4x4 -s ${param1} ${param2}")
GP_OPT_RESLICE_IMAGE=

OUTDIR=${ROOT}/out/${MODEL_DATE}_${param1}_${param2}_test
ATTN_DIR=${OUTDIR}/${MODEL_DATE}_test_${CHECK_POINT}_reg_${param1}_${param2}_attn
# ATTN_DIR=${ROOT}/out/${MODEL_DATE}_test_${CHECK_POINT}_reg_${param1}_${param2}
LOGDIR=${OUTDIR}/log
DUMPDIR=${OUTDIR}/dump
mkdir -p ${OUTDIR} ${LOGDIR} ${DUMPDIR} ${ATTN_DIR}

FIXED_LEFT=${ROOT}/reference/refspace_mprage_left.nii.gz 
FIXED_RIGHT=${ROOT}/reference/refspace_mprage_right.nii.gz
#FIXED_RIGHT=/project/hippogang_2/longxie/ASHS_T1/ASHSexp/exp201/atlas/final/template/refspace_mprage_right.nii.gz 

# Run stuff in queue
function pybatch()
{
  bash "$SCRIPT_DIRECTORY/pybatch.sh" -o "$DUMPDIR" "$@"
}


function runlog()
{
  local logfile CMD

  logfile=${1?}
  shift

  echo "$@" > "$logfile"
  CMD=${1?}
  shift

  $CMD "$@" | tee -a "$logfile"
}

function pairwise_registration()
{
 
  # read parameters for registration: moving_image

  MOVING=${1?}
  ATTENTION=${MOVING/.nii.gz/_attention.nii.gz}
  POSITION_MASK=${MOVING/.nii.gz/_position_mask.nii.gz}
  FILE_NAME=$(echo ${MOVING} | cut -d'/' -f 9 | cut -d'.' -f 1)
  echo $FILE_NAME

  if grep -q "left" <<< "$MOVING"; then
    FIXED=${FIXED_LEFT}
  else
    FIXED=${FIXED_RIGHT}
  fi

  echo $FIXED

  SUBJ_RESLICE_ATTENTION_OUTPUT=${ATTN_DIR}/${FILE_NAME}_reslice_attention.txt
  SUBJ_RESLICE_POSITION_MASK_OUTPUT=${ATTN_DIR}/${FILE_NAME}_reslice_position_mask.txt

  # registration intermediate outputs
  SUBJ_RIGID_MATRIX=${LOGDIR}/${FILE_NAME}_rigid.mat
  SUBJ_ROOT_WARP=${OUTDIR}/${FILE_NAME}_warp_root.nii.gz
  SUBJ_WARP=${OUTDIR}/${FILE_NAME}_warp.nii.gz

  SUBJ_AFFINE_MATRIX=${LOGDIR}/${FILE_NAME}_affine.mat
  SUBJ_RESLICE_ATTENTION_IMAGE=${ATTN_DIR}/${FILE_NAME}_reslice_attention.nii.gz
  SUBJ_RESLICE_POSITION_MASK_IMAGE=${ATTN_DIR}/${FILE_NAME}_reslice_position_mask.nii.gz

  # transform the attention map to the template space

  runlog "$SUBJ_RESLICE_ATTENTION_OUTPUT" greedy -d $GP_DIM -rf $FIXED -ri NN \
      $GP_OPT_RESLICE_IMAGE -rm $ATTENTION $SUBJ_RESLICE_ATTENTION_IMAGE \
      $MASK_RESLICE_CMD \
      -r \
      $([ $SUBJ_WARP ] && echo "$SUBJ_WARP") \
      $([ $SUBJ_AFFINE_MATRIX ] && echo "$SUBJ_AFFINE_MATRIX")


  runlog "$SUBJ_RESLICE_POSITION_MASK_OUTPUT" greedy -d $GP_DIM -rf $FIXED -ri NN \
      $GP_OPT_RESLICE_IMAGE -rm $POSITION_MASK $SUBJ_RESLICE_POSITION_MASK_IMAGE \
      $MASK_RESLICE_CMD \
      -r \
      $([ $SUBJ_WARP ] && echo "$SUBJ_WARP") \
      $([ $SUBJ_AFFINE_MATRIX ] && echo "$SUBJ_AFFINE_MATRIX")

}

function main () 
{

  MOVINGs=$(ls -d -1 ${DATADIR}/*_to_hw.nii.gz)
  for MOVING in $MOVINGs
  do
    FILE_NAME=$(echo ${MOVING} | cut -d'/' -f 9 | cut -d'.' -f 1)
    DUMP=${DUMPDIR}/${FILE_NAME}.txt

    echo $MOVING
    echo $DUMP
    # Compute the average
    pybatch -n 1 -N "${FILE_NAME}" -m 8G  ${ROOT}/script/step2_apply_attention_to_template.sh pairwise_registration ${MOVING}
    # pairwise_registration ${MOVING} 
  done

}



if [[ $1 ]]; then
  command=$1
  echo $1
  shift
  $command $@
else
    main
fi








