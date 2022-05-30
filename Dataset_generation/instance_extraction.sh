#!/bin/sh

#SBATCH --time=8-00
#SBATCH --begin=now
#SBATCH --gres=gpu:1
#SBATCH --job-name=generate_instances
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000M
#SBATCH --mail-type=end
#SBATCH --partition=gpu
#SBATCH --exclude=gpu06

module load anaconda/3-5.0.1
module load cuda/11.1
module load lib/cudnn/8.0.5.39_cuda_11.1
source activate predrnn
export IFN_DIR_DATASET=/beegfs/work/shared/
export IFN_DIR_CHECKPOINT=/beegfs/work/breitenstein/


srun python instance_extraction.py --datasetpath /beegfs/work/shared/cityscapes/ --savepath /beegfs/work/breitenstein/instance_data_full/
