import numpy as np
from PIL import Image
import os
from torchvision.transforms import Resize
import sys
import random
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/dataloader/')#path to IfN dataloader
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/')
from pt_data_loader.specialdatasets import CityscapesDataset, SimpleDataset
from definitions.labels_file import labels_cityscape_seg
import pt_data_loader.mytransforms as mytransforms
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor, ToPILImage
import json
import argparse
from os import path
import time

from cp_method import copy_paste
from cp_utils import get_street_horizon, trainid2rgb, vis_image_mask
import torch
import matplotlib.pyplot as plt


def filter_sourceimages(source_images):
    for i in source_images:
        image_src_rgba = np.array(Image.open(i))
        if image_src_rgba.shape[0]<10 or image_src_rgba.shape[1]<10:
            source_images.remove(i)
    return source_images
######################################################################

def generate_cp_dataset(instance_root, split, savepath, path_to_data, height, width, onlyval=True, filemode=False, pastefile = None):
    #we expect the instance to be split in train, val, test data
    labels_cs = labels_cityscape_seg.getlabels()
    print('split is', split)

    if 'test' in split:
        split_cs = 'validation'
    else:
        split_cs = 'train'
    print('split_cs is', split_cs)
    source_instances = [instance_root + '/' + split + '/' + f for f in os.listdir(instance_root + '/' + split) if f.endswith('.png')]

    source_instances = filter_sourceimages(source_instances)#filter for too small instances
    all_im_files = []
    for path1, subdirs, files in os.walk(path_to_data + split_cs):
        for name in files:
            all_im_files.append(name)

    #select a random 75 images for our validation dataset
    #we use two options for now
    #option 1: we select for the val dataset only instances from the 75 val images
    #option 2: we select for the val dataset instances from the 75 val images
    # and 1/10 of the training dataset (onlyval = False)
    print('check the source instances', source_instances[0])
    if 'test' in split:
        if not filemode:
            train_list = all_im_files
            val_list = []

        else:
            f = open(pastefile + '/test/' + 'pasting_dict_train.json')
            train_dict_load = json.load(f)
            val_list = []
            train_list = list(train_dict_load.keys())
            im_list_val = []

    else:
        if not filemode:
            val_list = random.sample(all_im_files,75)
            source_instances_wo_val = source_instances.copy()
            source_instances_val = []
            for val in val_list:
                all_im_files.remove(val)
                val0 = val.split('_left')[0]


                source_instances_val.extend([f for f in source_instances if val0 in f])
                source_instances_wo_val = [f for f in source_instances_wo_val if not val0 in f]
            train_list = all_im_files
            if not onlyval:
                additional_inst = random.sample(all_im_files, 290)
                for val in additional_inst:
                    val0 = val.split('.')[0]
                    source_instances_val.extend([f for f in source_instances if val0 in f])
                    source_instances_wo_val = [f for f in source_instances_wo_val if not val0 in f]

        else:
            f_val = open(pastefile + '/val/' + 'pasting_dict_val.json')
            f_train = open(pastefile + '/train/' + 'pasting_dict_train.json')
            val_dict_load = json.load(f_val)#load the dict
            train_dict_load = json.load(f_train)

            val_list = list(val_dict_load.keys()) #this contains just image names
            train_list = list(train_dict_load.keys())



    train_data_transforms = [mytransforms.CreateScaledImage(),
                             mytransforms.Resize((height, width)),
                             mytransforms.ConvertSegmentation(),  # convert the labels to their train IDs
                             mytransforms.CreateColoraug(new_element=True),
                             mytransforms.RemoveOriginals(),  # remove the images at native scale to save loading time
                             mytransforms.ToTensor(),
                             mytransforms.Relabel(255, 19),
                             ]

    train_dataset = CityscapesDataset(dataset='cityscapes',
                                  trainvaltest_split=split_cs,
                                  video_mode='mono',
                                  stereo_mode='mono',
                                  labels=labels_cs,
                                  split=None,
                                  labels_mode='fromtrainid',
                                  keys_to_load=['color', 'segmentation_trainid'],
                                  output_filenames=True,
                                  data_transforms=train_data_transforms)  #

    loader = DataLoader(train_dataset, 1, True, num_workers=1, pin_memory=True, drop_last=True)
    if 'test' in split:
        if filemode:
            cp_function_train = copy_paste(source_instances,
                                           occlusion_level=[0, 0.1],
                                           files=train_dict_load,
                                           filemode=True,
                                           gaussian_blurring=True, histogram_matching=True,
                                           coords='/beegfs/work/breitenstein/instance_data_full/test/coords.json')
        else:
            cp_function_train = copy_paste(source_instances,
                                     occlusion_level=[0, 0.1],
                                           gaussian_blurring = True, histogram_matching=True ,coords = '/beegfs/work/breitenstein/instance_data_full/test/coords.json')
        if not filemode:#if we generate a new dataset, we need a new empty dictionary
            train_pasting_dict = {}
        else:#if we generate from a file list then the dictionary is given
            train_pasting_dict = train_list
        print('length of instances', len(source_instances))
    else:
        if filemode:
            cp_function_train = copy_paste(source_instances,
                                           occlusion_level=[0, 0.1],
                                           files=train_dict_load,
                                           filemode=True,
                                           gaussian_blurring=True, histogram_matching=True,
                                           coords='/beegfs/work/breitenstein/instance_data_full/test/coords.json')
            cp_function_val = copy_paste(source_instances,
                                         files=val_dict_load,
                                         filemode=True,
                                         occlusion_level=[0, 0.1], gaussian_blurring = True, histogram_matching=True)
        else:
            cp_function_val = copy_paste(source_instances_val,
                                         occlusion_level=[0, 0.1], gaussian_blurring = True, histogram_matching=True)
            cp_function_train = copy_paste(source_instances_wo_val,
                                     occlusion_level=[0, 0.1], gaussian_blurring = True, histogram_matching=True)

            print('length of training instances', len(source_instances_wo_val))
            print('length of validation instances', len(source_instances_val))
        print('length of the validation list', len(val_list))
        if not filemode:
            val_pasting_dict = {}
            train_pasting_dict = {}
            print('the val list ist', val_list)
        else:#given dictionaries for file list
            train_pasting_dict = train_list
            val_pasting_dict = val_list
            print('the val list ist', val_list)



    for step, data in enumerate(loader):
        print('step is', step)
        image = data['color_aug',0,0]
        labels = data['segmentation_trainid', 0, 0]
        image = image[0].permute(1,2,0)
        labels = labels[0]
        files = data[('filename')]
        filename = files[('color', 0, -1)]

        if filemode:#sanity check
            print('filename is', filename[0])
            print('filename in list is ', train_list[0])
            if filename[0] in val_list:
                cur_split = 'val'
                cp_function = cp_function_val
            else:
                cur_split = split  # 'train'
                cp_function = cp_function_train
        else:
            if filename[0].split('/')[-1] in val_list:
                print('filename is in val_list')
                cur_split = 'val'
                cp_function = cp_function_val
            else:
                cur_split = split#'train'
                cp_function = cp_function_train

        print('cur split is', cur_split)



        #street_horizon = get_street_horizon(labels[0].numpy())
        starttime = time.time()
        result_image, bboxes_coords, semantic_mask, instance_mask, tgt_list = cp_function(image,None,filename)
        fintime = time.time()
        print('time for cp', fintime-starttime)
        print(filename[0])
        if not filemode:
            if cur_split == 'val':
                val_pasting_dict[filename[0]]=tgt_list
            else:
                train_pasting_dict[filename[0]]=tgt_list
        final_mask = torch.zeros(labels.shape[1], labels.shape[2],2)
        final_mask = final_mask.float()
        final_mask[:,:,1]=-1
        co = torch.where(semantic_mask != 0)

        final_mask[:,:,0] = labels[0]
        final_mask[co[0], co[1], 0] = semantic_mask.float()[semantic_mask != 0]
        final_mask[co[0], co[1], 1] = labels[0,semantic_mask != 0]

        vis_mask = torch.zeros(labels.shape[1], labels.shape[2])
        vis_mask[semantic_mask != 0]=1

        if not path.exists(savepath):
            os.mkdir(savepath)
        if not path.exists(savepath + '/' + cur_split):
            os.mkdir(savepath + '/' + cur_split)
            if not path.exists(savepath + '/' + 'val'):
                os.mkdir(savepath + '/' + 'val')
        if not path.exists(savepath + '/' + cur_split+ '/images/' ):
            os.mkdir(savepath + '/' + cur_split+ '/images/' )
        if not path.exists(savepath + '/' + cur_split+ '/labels/'):
            os.mkdir(savepath + '/' + cur_split+ '/labels/')


        plt.imsave(savepath + '/'+ cur_split + '/images/' + filename[0].split('/')[3:][0], np.array(result_image))
        vis_image_mask(result_image, vis_mask,savepath + '/' + cur_split + '/images/' + 'vis_image' + filename[0].split('/')[3:][0])
        plt.imsave(savepath + '/'+  cur_split + '/labels/' + filename[0].split('/')[3:][0].split('.')[0] + '_modal' + '.png', trainid2rgb(np.array(final_mask[:,:,0])).numpy())
        plt.imsave(savepath + '/'+ cur_split + '/labels/' + filename[0].split('/')[3:][0].split('.')[0] + '_amodal' + '.png', trainid2rgb(np.array(final_mask[:,:,1])).numpy())
        torch.save(final_mask, savepath + '/'+ cur_split + '/labels/' + filename[0].split('/')[3:][0].split('.')[0] + '.pt')

        json.dump(train_pasting_dict, open(savepath + '/' + cur_split + '/' + "pasting_dict_train.json", 'w'))
        if 'test' not in split:
            json.dump(val_pasting_dict, open(savepath + '/' + 'val/' + "pasting_dict_val.json", 'w'))

def main(args):
    split = args.split

    instanceroot = args.instanceroot
    generate_cp_dataset(instanceroot, split, args.savepath, args.datasetpath, args.height, args.width, filemode=args.filemode, pastefile = args.pastefile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath',
                        default = '/beegfs/work/shared/cityscapes/leftImg8bit/',
                        help='path to the Cityscapes dataset')
    parser.add_argument('--instanceroot',
                        default = '/beegfs/work/breitenstein/instance_data_full/',
                        help='path to the extracted instances')
    parser.add_argument('--split',
                        default = 'train',
                        help='split to create, possible is train and test, '
                             'train creates both training and validation split')
    parser.add_argument('--savepath',
                        default = '/beegfs/data/shared/amodalCS',
                        help='path where we save the new dataset')
    parser.add_argument('--height', type=int, default=1024,
                        help='height of the target images')
    parser.add_argument('--width', type=int, default=2048,
                        help='width of the target images')
    parser.add_argument('--filemode', default=False,
                        help='True means we generate our dataset from a json file')
    parser.add_argument('--pastefile', default = None,
                        help = 'path to the file that contains target images'
                               'with list of pasted instances and coordinates, '
                               'only for filemode True')
    args = parser.parse_args()

    main(args)