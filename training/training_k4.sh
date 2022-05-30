#!/bin/sh

#SBATCH --time=8-00
#SBATCH --begin=now
#SBATCH --gres=gpu:1
#SBATCH --job-name=amodaltrain
#SBATCH --exclude=gpu05
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000M
#SBATCH --mail-type=end
#SBATCH --partition=gpu

module load anaconda/3-5.0.1
module load cuda/10.0
module load lib/cudnn/7.3.1.20_cuda_10.0
source activate predrnn
export IFN_DIR_DATASET=/beegfs/work/shared/
export IFN_DIR_CHECKPOINT=/beegfs/work/breitenstein/


srun python training_k4.py --num-epochs 120 \
         --savedir amodal_cityscapes_K4\
         --decoder \
         --validation \
         --jobid ${SLURM_JOBID} \
         --dataroot AmodalCityscapes \
         --folderroot  /beegfs/work/breitenstein/ \
         --experimentname amodal_training_K4
