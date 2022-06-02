#!/bin/bash
#
#SBATCH --job-name=pada-stab2018-eval
#SBATCH --output=/ukp-storage-1/beck/slurm_output/pada-stab2018-eval
#SBATCH --mail-user=beck@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=turtok

. "/ukp-storage-1/beck/miniconda3.8/etc/profile.d/conda.sh"
conda activate pada
### Environment Variables
GPU_ID=0
CWD=`pwd`
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${CWD}

### Hyperparameters
CKPT_PATH=${CWD}/runs
TASK=stab2018
EVAL_BATCH_SIZE=32
EPOCHS=5
SEED=41

DOMAINS=(abortion cloning deathpenalty guncontrol marijuanalegalization minimumwage nuclearenergy schooluniforms)
for i in "${!DOMAINS[@]}"
do
  TRG_DOMAIN=${DOMAINS[$i]}
  DOMAINS_TMP=("${DOMAINS[@]}")
  unset DOMAINS_TMP[$i]
  SRC_DOMAINS=$(echo ${DOMAINS_TMP[*]} | tr ' ' ',')

  echo "Running evaluation checkpoint for $SRC_DOMAINS as source domains and $TRG_DOMAIN as target domain"

  set -x

  if [ ! -d "${CKPT_PATH}/${TASK}/${TRG_DOMAIN}/prompt_annotations" ]
  then
    echo "Extracting DRFs for the current experiment."
    python /ukp-storage-1/beck/Repositories/PADA/src/utils/drf_extraction.py \
    --domains ${SRC_DOMAINS} \
    --dtype ${TASK} \
    --drf_set_location /ukp-storage-1/beck/Repositories/PADA/runs/${TASK}/${TRG_DOMAIN}/drf_sets

    echo "Annotating training examples with DRF-based prompts."
    python /ukp-storage-1/beck/Repositories/PADA/src/utils/prompt_annotation.py \
    --domains ${SRC_DOMAINS} \
    --root_data_dir ${TASK}_data \
    --drf_set_location /ukp-storage-1/beck/Repositories/PADA/runs/${TASK}/${TRG_DOMAIN}/drf_sets \
    --prompts_data_dir /ukp-storage-1/beck/Repositories/PADA/runs/${TASK}/${TRG_DOMAIN}/prompt_annotations
  fi

  python /ukp-storage-1/beck/Repositories/PADA/eval.py \
  --dataset_name ${TASK} \
  --src_domains ${SRC_DOMAINS} \
  --trg_domain ${TRG_DOMAIN} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --seed ${SEED} \
  --ckpt_path /ukp-storage-1/beck/Repositories/PADA/runs/${TASK}/${TRG_DOMAIN}/PADA/e${EPOCHS}/b${EVAL_BATCH_SIZE}/a0.2/best_dev_macro_f1.ckpt \
  --output_dir /ukp-storage-1/beck/Repositories/PADA/runs/${TASK}_target_domain_label/${TRG_DOMAIN}/ \
  --replace_domain_label
done
