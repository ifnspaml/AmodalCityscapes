import numpy as np
import random
import sys
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/dataloader/')
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/')
from definitions.labels_file import labels_cityscape_seg
import PIL.Image as pil
import torch
import torchvision.transforms as transforms
import json
from scipy.ndimage import gaussian_filter
from skimage.transform import match_histograms


labels_cs = labels_cityscape_seg.getlabels()
Id2trainId = {label.id: label.trainId for label in reversed(labels_cs)}


class copy_paste(object):
    '''
    source_images - list of paths to source images (instances to paste in target image)
    occlusion_level - percentage value range (range 0,1) of the occlusion levels per image
    histogram_matching - apply histogram matching
    street_horizons - can be used to only choose instances with a similar street horizon as the current image (not implemented)
    files - path to file json to re-create an amodal dataset
    file_mode - True if we want to re-create an amodal dataset
    minimal_shape - minimal shape of the pasted instances
    allowed_overlap - how much are the pasted instances allowed to overlap with previously pasted ones
    gaussian_blurring - True to blur edges of the inserted instance (to make sure we do not learn sharp edges)


    '''

    def __init__(self, source_instances,
                 street_horizons = '/beegfs/work/breitenstein/Code/cv_repository/corner_case_detection/amodal_semantic_segmentation/dir_levels.json',
                 coords = '/beegfs/work/breitenstein/instance_data_full/train/coords.json',
                 occlusion_level = [0,0.25],
                 files=None,
                 filemode = False,
                 histogram_matching=False,
                 allowed_overlap = 0.0,
                 minimal_shape = (5,10),
                 gaussian_blurring=False):

        self.source_instances = source_instances

        self.occlusion_level = occlusion_level
        self.street_horizons = json.load(open(street_horizons))
        self.coords = json.load(open(coords))
        self.files = files
        self.filemode = filemode
        self.height = 1024
        self.width = 2048 #change if not Cityscapes


        self.allowed_overlap = allowed_overlap #percentage that the pasted instance can overlap with previously inserted instances
        self.minimal_shape = minimal_shape #minimal shape that we require for inserted instances

        self.gaussian_blurring = gaussian_blurring


        self.histogram_matching = histogram_matching

    def __call__(self, trgt_image, street_horizon=None, filename = None):#call to insert instances into a target image
        return self.insert_instances(trgt_image, street_horizon=street_horizon, filename=filename)


        #insert instances into the target image
    def insert_instances(self, trgt_image, street_horizon=None, filename = None):
        #trgt image is the image where we insert the instances
        #street horizon is the street horizon of the target image
        # filename of the target image, needed for filemode


        # #first treat filemode, we do not need to check anything but just paste the instances into the image


        tgt_list= []

        double_check_mask = torch.zeros((trgt_image.shape[0], trgt_image.shape[1]))
        idx = 0

        list_coords = []
        sem_seg_mask = torch.zeros((self.height, self.width), dtype=torch.int)  # *(-1)
        ins_seg_mask = torch.zeros((self.height, self.width), dtype=torch.int)  # *(-1)

        coverage_mask = torch.zeros((trgt_image.shape[0], trgt_image.shape[1]))

        oc_level = random.uniform(*self.occlusion_level) #sample an occlusion level from the predefined range
        coverage = 0 #we start with a coverage of 0

        if self.files is not None:#this treats the filemode
            # self.files is a dict if it exists. Then we get per target image the instances to paste
            print(self.files[filename[0]])
            paste_instances = self.files[filename[0]]
        else:
            paste_instances = [f for f in self.source_instances if not filename[0].split('/')[-1].split('_left')[0] in f]


        if self.filemode: #no while loop necessary for the occlusion level, we can just insert the instances w/o check
            counter=0
            for instance_full in paste_instances:
                inst_coords = instance_full[1]#
                instance = instance_full[0]
                instance_id = self.source_instances.index(instance)
                image_src, mask_src, inst_id, paste_coords, instance_name = self.get_src_instance(instance_id,
                                                                                              street_horizon)
                if inst_id < 1000:
                    sem_id = Id2trainId[inst_id]
                else:
                    sem_id = Id2trainId[inst_id // 1000]

                src_h, src_w, _ = image_src.shape
                x1, y1, x2, y2 = inst_coords

                x_diff = 0
                y_diff = 0
                if self.filemode:
                    if x1 == 0:
                        mask_x1 = -(x2 - src_w)
                        x_diff = -mask_x1
                    else:
                        x_diff = x1
                    if y1 == 0:
                        mask_y1 = -(y2 - src_h)
                        y_diff = -mask_y1
                    else:
                        y_diff = y1


                # histogram matching
                if self.histogram_matching:
                    image_src = self.histogram_match_function(trgt_image,x_diff,y_diff, image_src)


                #get paste mask and trgt_image with occlusions
                trgt_image, coords, mask = self.src_instance_paste(trgt_image, image_src, mask_src, x_diff, y_diff,
                                                             paste_coords,
                                                             coverage_mask)
                tgt_list.append((instance_name, inst_coords))

                #create the corresponding sem seg and instance seg masks
                inst_mask_semseg = torch.tensor(mask[y1:y2,x1:x2].clone()).type(torch.uint8) * (sem_id)


                sem_seg_mask[y1:y2,x1:x2]= inst_mask_semseg
                ins_seg_mask[y1:y2,x1:x2] += torch.tensor(mask[y1:y2,x1:x2].clone()).type(torch.int) * (inst_id)
                list_coords.append(inst_coords)
                counter+=1

        else:
            while coverage <= oc_level:#while the desired occlusion level for this target image is not reached


                if self.files is not None:
                    instance_id = self.source_instances.index(paste_instances[idx])#select the instance based on the filename
                else:
                    instance_id = self.source_instances.index(paste_instances[random.randint(0, len(paste_instances)-1)])

                    #randomly select an instance (just not from target image)

                image_src, mask_src, inst_id, paste_coords, instance_name = self.src_instance_choice(instance_id, street_horizon)


                if inst_id < 1000:
                    sem_id = Id2trainId[inst_id]
                else:
                    sem_id = Id2trainId[inst_id // 1000]


                #already calculate the coords here
                y1 = paste_coords[0][0]
                y_end = y1 + image_src.shape[0]

                #CARFEUL only uncomment if random selection of y_coordinates is also desired
                #y_end = np.random.randint(0,1024)

                #only use "free" coordinates to have higher chances that we can find a pasting location
                double_check_array = coverage_mask[y1, :]
                possible_xcoords = np.where(double_check_array == 0)[0]

                #CARFUL only uncomment if you want to use the same x-coordinate for pasting


                x_end = int(random.choice(possible_xcoords))

                #histogram matching
                if self.histogram_matching:
                    image_src = self.histogram_match_function(trgt_image,x_end,y_end, image_src)


                #get target image with pasted occluders, corresponding coordinates and the mask
                trgt_image, coords, mask = self.src_instance_paste(trgt_image, image_src, mask_src, x_end, y_end,paste_coords,
                                                            coverage_mask)


                if coords:#if coords for pasting were found
                    #obtain sem seg and instance seg masks
                    coverage_mask[mask != 0] = 1
                    coverage = torch.sum(coverage_mask) / (coverage_mask.shape[0] * coverage_mask.shape[1])
                    x1, y1, x2, y2 = coords

                    inst_mask_semseg = torch.tensor(mask[y1:y2, x1:x2].clone()).type(torch.uint8) * (sem_id)


                    img_sem_cp = sem_seg_mask[y1:y2, x1:x2].clone()
                    img_sem_cp[mask[y1:y2, x1:x2] == 1] = 0
                    img_ins_cp = ins_seg_mask[y1:y2, x1:x2].clone()
                    img_ins_cp[mask[y1:y2, x1:x2] == 1] = 0

                    ins_seg_trgt = torch.add(img_ins_cp, torch.tensor(mask[y1:y2, x1:x2].clone()).type(torch.uint8) * (inst_id))

                    sem_seg_mask[y1:y2, x1:x2] = inst_mask_semseg
                    ins_seg_mask[y1:y2, x1:x2] += ins_seg_trgt
                    tgt_ones = torch.add(img_sem_cp, inst_mask_semseg)
                    tgt_ones[tgt_ones != 0] = 1

                    double_check_mask[y1:y2, x1:x2] = tgt_ones
                    list_coords.append(coords)

                idx +=1




        coords_all = np.array(list_coords)

        return trgt_image, coords_all, sem_seg_mask, ins_seg_mask, tgt_list

    def histogram_match_function(self, trgt_image,x_diff,y_diff, image_src):
        reference = trgt_image[max(0, y_diff + image_src.shape[0] - 200):min(y_diff + image_src.shape[0] + 200, self.height),
                                max(0, x_diff + int(image_src.shape[1]/2) - 400):min(x_diff + int(image_src.shape[1]/2) + 400, self.width), :]
        image_src = torch.tensor(match_histograms(image_src[:, :, :3].numpy(), reference.numpy(), multichannel=True))
        return image_src


    def src_instance_paste(self, trgt_image, src_image, mask_src, x_end, y_end, paste_coords, double_check_mask):

        src_h, src_w, src_c = src_image.shape


        mask_visible = torch.zeros((self.height, self.width))#*(-1)


        x_diff = x_end - int(src_w/2)
        y_diff = y_end - src_h
        h1 = max(y_diff, 0)
        h2 = min(y_diff + src_h, self.height)
        w1 = max(x_diff, 0)
        w2 = min(x_diff + src_w, self.width)


        if h1 >0:
            h1_mask = 0
        else:
            h1_mask =-y_diff
        if w1 > 0:
            w1_mask = 0
        else:
            w1_mask = -x_diff
        if h2 < self.height - 1:
            h2_mask = src_h
        else:
            h2_mask = self.height -y_diff
        if w2 < self.width - 1:
            w2_mask = src_w
        else:
            w2_mask = self.width -x_diff


        coords = []
        #check for overlap
        if torch.sum(double_check_mask[h1:h2, w1:w2])/(double_check_mask[h1:h2, w1:w2].shape[0]*double_check_mask[h1:h2, w1:w2].shape[1]) > self.allowed_overlap:
            #print('return because double check fails:', torch.sum(double_check_mask[y1:y2, x1:x2])/(double_check_mask[y1:y2, x1:x2].shape[0]*double_check_mask[y1:y2, x1:x2].shape[1]))
            return trgt_image, coords, None


        if h1_mask >= src_h or w1_mask >= src_w or h2_mask < 0 or w2_mask < 0:
            return trgt_image, coords, None



        rgb_values = src_image.clone()


        occluded_area = trgt_image[h1:h2, w1:w2].clone()

        visible_area = rgb_values[h1_mask:h2_mask, w1_mask:w2_mask].clone()


        if self.gaussian_blurring:
            gaussian_blur = torch.tensor(gaussian_filter(mask_src[h1_mask:h2_mask, w1_mask:w2_mask].clone().float(),sigma=1))#np.sqrt(2)))
            #with gaussian blur
            visible_rgb = occluded_area.permute(2,0,1)*(1-gaussian_blur) + visible_area.permute(2,0,1)*gaussian_blur
            visible_rgb = visible_rgb.permute(1,2,0)

        #without gaussian blur
        else:
            occluded_area[mask_src[h1_mask:h2_mask, w1_mask:w2_mask] == 1, :] = 0 #set the target image to zero where the instance is pasted
            visible_area[mask_src[h1_mask:h2_mask, w1_mask:w2_mask] < 1, :] = 0 #set the source image to zero outside the instance mask
            visible_rgb = torch.add(occluded_area, visible_area)


        mask_visible[h1:h2, w1:w2] = mask_src[h1_mask:h2_mask, w1_mask:w2_mask]
        trgt_image[h1:h2, w1:w2] = visible_rgb  # insert the visible
        coords = [w1, h1, w2, h2]


        return trgt_image, coords, mask_visible

    def get_src_instance(self, idx, street_horizon = None):
        if street_horizon!= None:
            im_list = self.street_horizons[street_horizon]
            instance_list = []
            for im in im_list:
                cur_list = [f for f in self.source_instances if im in f]
                instance_list.extend(cur_list)

            object_idx = random.randint(0, len(instance_list) - 1)

            path_to_src_instance = instance_list[object_idx]
        else:
            path_to_src_instance = self.source_instances[idx]
        image_src_rgba = np.array(pil.open(str(path_to_src_instance)))
        inst_id = str(path_to_src_instance).split('_')[-1].split('.')[0]


        image_src = image_src_rgba[:, :, :3]
        mask_src = image_src_rgba[:, :, 3]
        image_src = pil.fromarray(image_src.astype('uint8'))
        image_src = transforms.ToTensor()(image_src)

        paste_coords = self.coords[str(path_to_src_instance).split('/')[-1].split('.')[0]]
        return image_src.permute(1, 2, 0), torch.tensor(mask_src), int(inst_id), paste_coords, str(path_to_src_instance)
