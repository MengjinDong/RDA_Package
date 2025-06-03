#!/bin/bash
set -x -e

# This script tries to find the correct parameter setting for a pairwise registration 
# from a bl or fu image to a template image

# In the test stage, images will only be normalized and cropped, but not flipped
# left images are registered to the left template; 
# right images are registered to the right one.


SCRIPT_DIRECTORY=$(dirname "$0")

ROOT=/project/hippogang_4/mdong/Longi_T1_Aim2/attention_heatmap
#MODEL_DATE="2022-04-29_17-14"
#CHECK_POINT="0040_crop"
MODEL_DATE="2023-12-06_13-28" # 2022-07-27_21-17, 0020
CHECK_POINT="0024"
DATADIR=${ROOT}/data/${MODEL_DATE}_val_${CHECK_POINT}   # model_name, which epoch of evaluation

GP_DIM=3

GP_OPT_RIGID="-n 100x50x40 -m WNCC 4x4x4"
GP_OPT_AFFINE="-n 100x50x40 -m WNCC 4x4x4"
param1=3mm
param2=0.05mm
GP_OPT_DEFORM=$(echo  "-n 100x50x40 -m WNCC 4x4x4 -s ${param1} ${param2}")
GP_OPT_RESLICE_IMAGE=

OUTDIR=${ROOT}/out/${MODEL_DATE}_${param1}_${param2}_test
LOGDIR=${OUTDIR}/log
DUMPDIR=${OUTDIR}/dump
ATTN_DIR=${OUTDIR}/${MODEL_DATE}_test_${CHECK_POINT}_reg_${param1}_${param2}_attn
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

  # specify where to save log files
  # mkdir -p ${OUTDIR}/${subjectID}
  SUBJ_RIGID_OUTPUT=${LOGDIR}/${FILE_NAME}_log_rigid.txt
  SUBJ_AFFINE_OUTPUT=${LOGDIR}/${FILE_NAME}_log_affine.txt
  SUBJ_DEFORM_OUTPUT=${LOGDIR}/${FILE_NAME}_log_deform.txt
  SUBJ_RESLICE_RIGID_OUTPUT=${LOGDIR}/${FILE_NAME}_log_reslice_rigid.txt
  SUBJ_RESLICE_AFFINE_OUTPUT=${LOGDIR}/${FILE_NAME}_log_reslice_affine.txt
  SUBJ_RESLICE_DEFORM_OUTPUT=${LOGDIR}/${FILE_NAME}_log_reslice_deform.txt
  SUBJ_RESLICE_ATTENTION_OUTPUT=${LOGDIR}/${FILE_NAME}_log_reslice_attention.txt
  SUBJ_RESLICE_POSITION_MASK_OUTPUT=${ATTN_DIR}/${FILE_NAME}_reslice_position_mask.txt

  # registration intermediate outputs
  SUBJ_RIGID_MATRIX=${LOGDIR}/${FILE_NAME}_rigid.mat
  SUBJ_ROOT_WARP=${OUTDIR}/${FILE_NAME}_warp_root.nii.gz
  SUBJ_WARP=${OUTDIR}/${FILE_NAME}_warp.nii.gz

  SUBJ_AFFINE_MATRIX=${LOGDIR}/${FILE_NAME}_affine.mat
  SUBJ_RESLICE_RIGID_IMAGE=${OUTDIR}/${FILE_NAME}_reslice_rigid.nii.gz
  SUBJ_RESLICE_AFFINE_IMAGE=${OUTDIR}/${FILE_NAME}_reslice_affine.nii.gz
  SUBJ_RESLICE_DEFORM_IMAGE=${OUTDIR}/${FILE_NAME}_reslice_deform.nii.gz
  SUBJ_RESLICE_ATTENTION_IMAGE=${ATTN_DIR}/${FILE_NAME}_reslice_attention.nii.gz
  SUBJ_RESLICE_POSITION_MASK_IMAGE=${ATTN_DIR}/${FILE_NAME}_reslice_position_mask.nii.gz

  # greedy registration: first rigid, then affine, then deformable registration
  # register all images to the template space
  # rigid registration

  runlog "$SUBJ_RIGID_OUTPUT" greedy -d $GP_DIM -a -dof 6 \
        -i $FIXED $MOVING \
        $MASK_CMD $GP_OPT_RIGID \
	-ia-image-centers \
        -o $SUBJ_RIGID_MATRIX

#        $([ -f $SUBJ_RIGID_MATRIX ] && echo "-ia $SUBJ_RIGID_MATRIX") \


  # transformation after rigid registration

#  runlog "$SUBJ_RESLICE_RIGID_OUTPUT" greedy -d $GP_DIM -rf $FIXED \
#      $GP_OPT_RESLICE_IMAGE -rm $MOVING $SUBJ_RESLICE_RIGID_IMAGE \
#      $MASK_RESLICE_CMD \
#      -r \
#      $([ $SUBJ_RIGID_MATRIX ] && echo "$SUBJ_RIGID_MATRIX")

  # affine registration

  runlog "$SUBJ_AFFINE_OUTPUT" greedy -d $GP_DIM -a -dof 12 \
          -i $FIXED $MOVING \
          $MASK_CMD $GP_OPT_AFFINE \
          $([ $SUBJ_RIGID_MATRIX ] && echo "-ia $SUBJ_RIGID_MATRIX") \
          -o $SUBJ_AFFINE_MATRIX

  # transformation after affine registration
 
#  runlog "$SUBJ_RESLICE_AFFINE_OUTPUT" greedy -d $GP_DIM -rf $FIXED \
#      $GP_OPT_RESLICE_IMAGE -rm $MOVING $SUBJ_RESLICE_AFFINE_IMAGE \
#      $MASK_RESLICE_CMD \
#      -r \
#      $([ $SUBJ_AFFINE_MATRIX ] && echo "$SUBJ_AFFINE_MATRIX")


  # deformable registration

  runlog "$SUBJ_DEFORM_OUTPUT" greedy -d $GP_DIM -sv \
        -i $FIXED $MOVING \
        $MASK_CMD $GP_OPT_DEFORM \
        $([ $SUBJ_AFFINE_MATRIX ] && echo "-it $SUBJ_AFFINE_MATRIX") \
        -oroot $SUBJ_ROOT_WARP -o $SUBJ_WARP


  # transform moving image to the template space

#  runlog "$SUBJ_RESLICE_DEFORM_OUTPUT" greedy -d $GP_DIM -rf $FIXED \
#      $GP_OPT_RESLICE_IMAGE -rm $MOVING $SUBJ_RESLICE_DEFORM_IMAGE \
#      $MASK_RESLICE_CMD \
#      -r \
#      $([ $SUBJ_WARP ] && echo "$SUBJ_WARP") \
#      $([ $SUBJ_AFFINE_MATRIX ] && echo "$SUBJ_AFFINE_MATRIX")

  # transform the attention map to the template space

  runlog "$SUBJ_RESLICE_ATTENTION_OUTPUT" greedy -d $GP_DIM -rf $FIXED -ri NN \
      $GP_OPT_RESLICE_IMAGE -rm $ATTENTION $SUBJ_RESLICE_ATTENTION_IMAGE \
      $MASK_RESLICE_CMD \
      -r \
      $([ $SUBJ_WARP ] && echo "$SUBJ_WARP") \
      $([ $SUBJ_AFFINE_MATRIX ] && echo "$SUBJ_AFFINE_MATRIX")

  # transform position mask to the template space

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
  #FILE_NAME=$(echo ${MOVING} | cut -d'/' -f 9 | cut -d'.' -f 1)
  #DUMP=${DUMPDIR}/${FILE_NAME}.txt
  for MOVING in $MOVINGs
  do
    echo $MOVING
    FILE_NAME=$(echo ${MOVING} | cut -d'/' -f 9 | cut -d'.' -f 1)
    DUMP=${DUMPDIR}/${FILE_NAME}.txt
    echo $DUMP
    if [ ! -f  ${OUTDIR}/${FILE_NAME}_warp.nii.gz ]; then
      # Compute the average
      pybatch -N "${FILE_NAME}" -n 1 -m 8G  ${ROOT}/script/step12_reg_to_temp_and_apply_attn_for_all.sh pairwise_registration ${MOVING}
      # pairwise_registration ${MOVING}
    fi
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








