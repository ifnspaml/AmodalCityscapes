import torch
import sys
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/dataloader/')
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/')
from definitions.labels_file import labels_cityscape_seg
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

labels_cs = labels_cityscape_seg.getlabels()

def id2trainId(labels):
    Id2label = {label.id: label for label in reversed(labels_cs)}  # e.g. trainId2label[index].name gives name of class
    new_label = labels.clone()
    for i in torch.unique(labels):
        new_label[labels == i] = Id2label[i.item()].trainId

    new_label[new_label == 255] = 19
    return new_label

#%% convert the train ids to rgb
def trainid2rgb(labels):
    print('labels shape in trainid2rgb', labels.shape)
    labels_cs = labels_cityscape_seg.getlabels()
    trainId2label   = { label.trainId : label for label in reversed(labels_cs) } #e.g. trainId2label[index].name gives name of class
    rgb_label = torch.zeros((labels.shape[0],labels.shape[1],3))
    print('rgb_label shape in trainid2rgb', rgb_label.shape)
    for i in range(0,19):
        rgb_label[labels==i,:]=torch.tensor(trainId2label[i].color,dtype=torch.float)/255

    rgb_label[labels==-1,:]=torch.tensor([255,255,255],dtype=torch.float)/255
    return rgb_label

#%% convert the rgb to trainids
def rgb2trainId(rgb_labels):
    labels_cs = labels_cityscape_seg.getlabels()
    trainId2label   = { label.trainId : label for label in reversed(labels_cs) } #e.g. trainId2label[index].name gives name of class
    id_label = torch.ones((rgb_labels.shape[0],rgb_labels.shape[1]))*(-1)
    print(torch.unique(rgb_labels))
    print(rgb_labels.shape)
    for i in range(0,19):
        rgb_val = trainId2label[i].color
        print(rgb_val)
        ind = np.where((rgb_labels[:, :, 0] == rgb_val[0]) & (rgb_labels[:, :, 1] == rgb_val[1]) & (rgb_labels[:, :, 2] == rgb_val[2]))
        id_label[ind]=i

    id_label[id_label==-1]=19
    return id_label
#%% visualization for image with inserted instances
def vis_image_mask(image, mask, filename):
    #image is an RGB image as torch tensor heightxwidthx3
    #mask is a torch tensor with elements in {0,1}, heightxwidth
    #we overlay the inserted instances from the mask in red
    overlay_mask = image.clone()
    overlay_mask[mask == 1] = 0.7 * overlay_mask[mask == 1] + 0.3 * torch.tensor([1, 0, 0])
    plt.imsave(filename + '.png', np.array(overlay_mask))

#%% visualization of masks (either amodal or modal)
def vis_mask(mask, filename):
    # mask is torch tensor with semantic labels of size heightxwidth
    label_color = trainid2rgb(mask)
    plt.imsave(filename + '.png', label_color)
    return label_color

#%% get the street horizon level per image
def street_horizon():
    path_to_cs = 'C:/Users/breitenstein/Documents/Datasets/cityscapes/gtFine/train/'
    dir_levels = {'0': [], '64': [], '128': [], '192': [], '256': [], '320': [], '384': [], '448': [], '512': [], '576': [],
           '640': [], '704': [], '768': [], '832': [], '896': [], '960': []}
    for folder in os.listdir(path_to_cs):
        files = os.listdir(path_to_cs + folder)
        for file in files:
            if 'color' in file:
                print(path_to_cs + '/' + folder + '/' + file)
                im = Image.open(path_to_cs + '/' + folder + '/' + file).convert('RGB')
                im_array = np.array(im)
                mask = np.zeros((im_array.shape[0], im_array.shape[1]))
                c = np.where(im_array == np.array([128, 64, 126]))
                mask[c[0], c[1]] = 1
                a = np.sum(mask, axis=1)
                total = np.sum(a)
                if total == 0:
                    dir_levels['960'].append(file.split('_g')[0])
                else:
                    w = 0.01 * total / 100
                    pixel = 0
                    step = 0
                    while pixel<=w:
                        pixel +=a[step]
                        step +=1
                    im_level = int((step//64)*64)
                    dir_levels[str(im_level)].append(file.split('_g')[0])

    return dir_levels

#%%
def get_street_horizon(sem_seg):
    mask = np.zeros((sem_seg.shape[0], sem_seg.shape[1]))
    c = np.where(sem_seg == 0)
    mask[c[0], c[1]] = 1
    a = np.sum(mask, axis=1)
    total = np.sum(a)
    if total == 0:
        return '960'
    else:
        w = 0.01 * total / 100
        pixel = 0
        step = 0
        while pixel <= w:
            pixel += a[step]
            step += 1
        im_level = int((step // 32) * 64) #workaround to get the correct level even though our image is smaller

    return str(im_level)