#!/bin/sh

#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=NormalEval
#SBATCH --exclude=gpu05
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000M
#SBATCH --mail-type=end
#SBATCH --partition=debug

export IFN_DIR_DATASET=/beegfs/work/shared/
export IFN_DIR_CHECKPOINT=/beegfs/work/breitenstein/

module load anaconda/3-5.0.1
module load cuda/10.0
module load lib/cudnn/7.3.1.20_cuda_10.0
source activate predrnn


srun python eval_iou_all.py \
--loadDir /beegfs/work/breitenstein/Code/cv_repository/ \
--loadWeights corner_case_detection/save/amodal_cityscapes_K3/model_best.pth  \
--loadModel corner_case_detection/save/ERFNet_ImagenetPretrain/erfnet \
--modeltype amodal --dataset amodal --numgroups 3 --numclasses 26 \
