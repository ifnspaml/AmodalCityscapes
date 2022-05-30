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
--loadWeights corner_case_detection/save/main_cityscapes_copy_paste_nooverlap_gaussianblur_originalsize_histogram_matching_standardERFNet_120/model_best.pth  \
--loadModel corner_case_detection/model/erfnet \
--plot True \
--plotpath  normalERFNet/CSVal/ \
--modeltype normal --dataset normal --numgroups 3 --numclasses 20 \

