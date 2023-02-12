#!/bin/bash
#
#SBATCH --job-name=t5-stab2018
#SBATCH --output=/ukp-storage-1/beck/slurm_output/t5-stab2018
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
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=`pwd`

### Hyperparameters
TASK=stab2018
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
EPOCHS=5
SEED=41
DATADIR=/ukp-storage-1/beck/Repositories/PADA/data
OUTPUT_DIR=/ukp-storage-1/beck/Repositories/PADA/runs/T5-topic

DOMAINS=(abortion cloning deathpenalty guncontrol marijuanalegalization minimumwage nuclearenergy schooluniforms)
for i in "${!DOMAINS[@]}"
do
  TRG_DOMAIN=${DOMAINS[$i]}
  DOMAINS_TMP=("${DOMAINS[@]}")
  unset DOMAINS_TMP[$i]
  SRC_DOMAINS=$(echo ${DOMAINS_TMP[*]} | tr ' ' ',')

  set -x
  echo "Running experiment for $SRC_DOMAINS as source domains and $TRG_DOMAIN as target domain"

  python /ukp-storage-1/beck/Repositories/PADA/train.py \
  --model_name T5 \
  --dataset_name ${TASK} \
  --src_domains ${SRC_DOMAINS} \
  --trg_domain ${TRG_DOMAIN} \
  --num_train_epochs ${EPOCHS} \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --seed ${SEED} \
  --add_domain \
  --output_dir ${OUTPUT_DIR}
done
