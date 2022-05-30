#!/bin/sh

#SBATCH --time=00:20:00
#SBATCH --begin=now
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1
#SBATCH --job-name=generate_instances
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=end
#SBATCH --partition=debug
#SBATCH --exclude=gpu04

module load anaconda/3-5.0.1
module load cuda/11.1
module load lib/cudnn/8.0.5.39_cuda_11.1
source activate predrnn
export IFN_DIR_DATASET=/beegfs/work/shared/
export IFN_DIR_CHECKPOINT=/beegfs/work/breitenstein/


srun python generate_cp_dataset.py --split test \
--pastefile /beegfs/work/breitenstein/AmodalCityscapes \
--filemode True


srun python generate_cp_dataset.py --split train \
--pastefile /beegfs/work/breitenstein/AmodalCityscapes \
--filemode True
