import numpy as np
from PIL import Image
import os
from torchvision.transforms import Resize
import sys
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/dataloader/')
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/')
from definitions.labels_file import labels_cityscape_seg
import json
import argparse
from os import path


def instance_extraction_cityscapes(image_path, labels_path, save_dir, height, width, data_split=['train','val']):
    class_names = ['person','rider','car','truck','bus','train','motorcycle','bicycle']
    labels_cs = labels_cityscape_seg.getlabels()

    name_to_id = { label.name : label.id for label in reversed(labels_cs) }


    coord_dict = {}


    for split in data_split:
        if split == 'val': #Cityscapes validation split is the amodal CS test split
            cp_split = 'test'
        else: cp_split = 'train'
        if not path.exists(save_dir + '/' + cp_split):
            os.mkdir(save_dir + '/' + cp_split)
        labels = labels_path + '/' + split
        images = image_path + '/' + split

        cities = os.listdir(images)

        for city in cities: #iterate through the city folders
            images_city = images + '/' + city
            labels_city = labels + '/' + city

            image_list = [f for f in os.listdir(images_city) if f.endswith('.png')]

            for image_str in image_list: #iterate through the images in that folder
                instancename = image_str.split('_l')[0] + '_gtFine_instanceIds.png' #load instances
                labelname = image_str.split('_l')[0] + '_gtFine_labelIds.png' #load segmentation labels

                image = Image.open(images_city + '/' + image_str)
                mask = Image.open(labels_city + '/' + labelname)
                instance = Image.open(labels_city + '/' + instancename)

                #resize to specified height and width of the dataset
                resize_image = Resize((height,width), interpolation=Image.BILINEAR) #possibility to resize if desired
                resize_masks = Resize((height,width), interpolation=Image.NEAREST)
                image = resize_image(image)
                image = np.array(image)
                mask = resize_masks(mask)
                mask = np.array(mask)
                instance = resize_masks(instance)
                instance = np.array(instance)

                for name in class_names: #go through the instance class names
                    class_id = name_to_id[name]#get the corresponding id

                    sem_mask = np.zeros((mask.shape))
                    sem_mask[mask==class_id] = 1 #semantic segmentation mask for the respective class with class_id

                    instance_mask = instance.copy()
                    instance_mask[sem_mask!=1]=0 #keep only instances belonging to the respective semantic class class_id

                    instance_ids = np.unique(instance_mask)
                    instance_ids = instance_ids[instance_ids != 0]#get unique instance ids

                    for instance_id in instance_ids:
                        cur_instance = np.zeros(instance_mask.shape)
                        cur_instance[instance_mask==instance_id]=1 #current instance mask

                        im_instance = image.copy()
                        #im_instance[cur_instance!=1]=0

                        mask_coords = np.where(cur_instance>0)
                        y_mask_coords = mask_coords[0]
                        x_mask_coords = mask_coords[1]

                        min_x, max_x = min(x_mask_coords), max(x_mask_coords)+1
                        min_y, max_y = min(y_mask_coords), max(y_mask_coords)+1

                        print([[int(min_y),int(max_y)],[int(min_x),int(max_x)]]) #the bounding box coords of the instance

                        #extract image in this area
                        extracted_instance_image = im_instance[min_y:max_y, min_x:max_x, :]
                        extracted_instance_mask = cur_instance[min_y:max_y, min_x:max_x]

                        size_not_zero = len(extracted_instance_mask[extracted_instance_mask>0]) \
                                        / extracted_instance_mask.shape[0] * extracted_instance_mask.shape[1]


                        if size_not_zero > 0.5: #if enough nonzero components
                            rgbainstance_image_path = save_dir + '/' + cp_split +'/' +  image_str.split('_l')[0]  + '_%s' %instance_id # city_image.name.replace(postfix_image,f'{instance_id}')

                            rgba_image = np.zeros((extracted_instance_image.shape[0],extracted_instance_image.shape[1],4),dtype ='uint8')
                            rgba_image[:,:,0:3]=extracted_instance_image #the saved rgba image contains the extracted image in the bounding box
                            rgba_image[:,:,3]=extracted_instance_mask# and the instance mask

                            rgba_pil = Image.fromarray(rgba_image,'RGBA')
                            rgba_pil.save(rgbainstance_image_path + '.png')

                            coord_dict[rgbainstance_image_path.split('/')[-1].split('.')[0]] = [[int(min_y),int(max_y)],[int(min_x),int(max_x)]] #coord dictionary contains the original coordinates

        json.dump(coord_dict, open(save_dir + '/' + cp_split +'/' + "coords.json", 'w'))


def main(args):
    image_path = args.datasetpath + '/leftImg8bit/'
    labels_path = args.datasetpath + '/gtFine/'

    if not path.exists(args.savepath):
        os.mkdir(args.savepath)

    instance_extraction_cityscapes(image_path, labels_path, args.savepath, args.height, args.width, data_split=['train', 'val'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', help='path to the dataset')
    parser.add_argument('--savepath', help = 'path where we store the extracted instances')
    parser.add_argument('--height', type=int , default=1024, help='path where we store the extracted instances')
    parser.add_argument('--width', type=int , default = 2048, help='path where we store the extracted instances')
    args = parser.parse_args()

    main(args)