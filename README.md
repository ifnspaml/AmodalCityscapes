# Amodal Cityscapes


This repository contains the code to re-generate our Amodal Cityscapes dataset.
For more details, please refer to our paper:

    @InProceedings{Breitenstein2022,
    author    = {Breitenstein, J. and Fingscheidt, T.},
    booktitle = {Proc. of IV},
    title     = {{Amodal Cityscapes: A New Dataset, its Generation, and an Amodal Semantic Segmentation Challenge Baseline}},
    year      = {2022},
    address   = {Aachen, Germany},
    month     = jun,
    pages     = {1--8},
    }

#### Installation

Requirements to run our code are:
- Pytorch 1.10.0
- [IfN dataloader](https://github.com/ifnspaml/IFN_Dataloader)
- Cuda 11.1
- TensorboardX (optional)

We provide both the environment.yml file and the requirements.txt. 
We recommend installing the environment from here for completeness.
You can install them using:


`conda env create -f environment.yml`

####Dataset Creation

This paragraph describes how to 

a) re-generate the Amodal Cityscapes dataset

b) generate your own copy-paste dataset

##### Instance  Extraction
To make dataset creation easier, we first extract the instances from the standard dataset. In our case, this is the Cityscapes dataset.

To run the instance extraction, run:

    python instance_extraction.py 
    --datasetpath [path to folder]/cityscapes/ 
    --savepath [path where you want to save the instances]


You can also use the provided SLURM script. The Python script assumes that labels and images are located in the standard Cityscapes way.
This needs to be run also for regeneration of the Amodal Cityscapes dataset.
If you need to generate a new dataset from a different base dataset, the code needs to be adapted to the desired filenames.

##### Dataset Generation

a) Re-Generate the Amodal Cityscapes dataset

For this run 

    python generate_cp_dataset.py 
    --split test \
    --pastefile [path to the folder with the dictionaries for pasting] \
    --filemode True
    --instanceroot [path where the source instances are saved]
    --datasetpath [path to the target images]

This assumes the pastefiles are located in a folder with `train`, `val`, and `test` subfolders.
Change the `--split` to `train` to re-generate the training and validation dataset split.
The `datasetpath` argument assumes that the Cityscapes images are located in the `leftImg8bit` folder as described by the standard dataset.

b) Generate your own amodal copy-paste dataset

For this run

    python generate_cp_dataset.py 
    --split test 
    --savepath [path to save the dataset to]
    --instanceroot [path where the source instances are saved]
    --datasetpath [path to the target images]
    
Run this with split `test` and `train` to generate a training, validation and test dataset split. Running with split `train` generates
the training and validation dataset split by choosing 75 randomly selected images of the Cityscapes training dataset as the validation split.
Per default, the script uses Gaussian blurring and histogram matching when inserting the source instances.

#### Training

Go to the training folder. 
We used the ERFNet for training [1]. You can find the modelfile in the respective [repository](https://github.com/Eromera/erfnet_pytorch). 
However you can also use any semantic segmentation model of your choice.
We adapted the method from Purkait et al. [2] to the setting described in our paper.

To run the training for K=4 groups, run
    
    python training_k4.py --num-epochs 120 \
         --imagenet-weight-path [path-to-weights-from-imagenet-pretraining] \
         --model [path to modelfile] \
         --savedir [directory for saving checkpoints] \
         --decoder \
         --validation \
         --jobid ${SLURM_JOBID} \
         --dataroot [folder of the amodal dataset] \
         --folderroot [path to the amodal dataset folder]  \
         --experimentname [name of the experiment for tensorboard]
         
Other settings for K=4 are set per default in the python file `training_k4.py` like the number of classes.

To run the training for K=3 groups, run
    
    python training_k3.py --num-epochs 120 \
         --imagenet-weight-path [path-to-weights-from-imagenet-pretraining] \
         --model [path to modelfile] \
         --savedir [directory for saving checkpoints] \
         --decoder \
         --validation \
         --jobid ${SLURM_JOBID} \
         --dataroot [folder of the amodal dataset] \
         --folderroot [path to the amodal dataset folder]  \
         --experimentname [name of the experiment for tensorboard]
         
Other settings for K=3 are set per default in the python file `training_k3.py` like the number of classes.

#### Evaluation

You can evaluate your trained models using the scripts in the Evaluation folder. 

There are three SLURM scripts to evaluate normal semantic segmentation methods, and the K=3/K=4 settings.
Otherwise for K=3, run:

    python eval_iou_all.py \
    --loadDir /beegfs/work/breitenstein/Code/cv_repository/ \
    --loadWeights corner_case_detection/save/amodal_cityscapes_K3/model_best.pth  \
    --loadModel corner_case_detection/save/ERFNet_ImagenetPretrain/erfnet \
    --modeltype amodal --dataset amodal --numgroups 3 --numclasses 26 \

For K=4 run:

    eval_iou_all.py \
    --loadDir /beegfs/work/breitenstein/Code/cv_repository/ \
    --loadWeights corner_case_detection/save/amodal_cityscapes_K4/model_best.pth  \
    --loadModel corner_case_detection/save/amodal_cityscapes_K4/erfnet \
    --modeltype amodal --dataset normal --numgroups 4 --numclasses 28 \

For the normal evaluation, run:
    
    python eval_iou_all.py \
    --loadDir /beegfs/work/breitenstein/Code/cv_repository/ \
    --loadWeights corner_case_detection/save/main_cityscapes_copy_paste_nooverlap_gaussianblur_originalsize_histogram_matching_standardERFNet_120/model_best.pth  \
    --loadModel corner_case_detection/model/erfnet \
    --modeltype normal --dataset normal --numgroups 3 --numclasses 20 \

#### References
[1] E. Romera, J. M. Álvarez, L. M. Bergasa, and R. Arroyo, “ERFNet:
Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation,”
IEEE Transactions on Intelligent Transportation Systems,
vol. 19, no. 1, pp. 263–272, Jan. 2018.

[2] P. Purkait, C. Zach, and I. D. Reid, “Seeing Behind Things: Extending
Semantic Segmentation to Occluded Regions,” in Proc. of IROS,
Macau, SAR, China, Nov. 2019, pp. 1998–2005.
