import os
import sys
import random
import numpy as np
import torch
os.environ['PYTHONHASHSEED'] = '0'
seed=1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
import json
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime, date
import time
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/corner_case_detection/model/')
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/dataloader/')
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/')
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/corner_case_detection/model/')

import torch.nn.functional as F

from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from definitions.labels_file import labels_cityscape_seg
import torchvision.transforms.functional as transforms_fun
from torchvision import datasets, transforms
from Evaluation.evaluation import evaluator

import random
import pt_data_loader.mytransforms as mytransforms
import importlib
from iouEval import iouEval, getColorEntry
from shutil import copyfile
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def get_paser_options():
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="./../erfnet", help='Path to model.py file')#path to the ERFNet file
    parser.add_argument('--non-deterministic', action='store_false', default=False)
    parser.add_argument('--height',
                        type=int,
                        default=512)
    parser.add_argument('--width',
                        type=int,
                        default=1024)
    parser.add_argument('--dataset',
                        type=str,
                        default='amodal',
                        help='Name of the dataset that will be learned')
    parser.add_argument('--modeltype',
                        type=str,
                        default='erfnet',
                        help='specify which model to load. Only ERFNet implemented so far')
    parser.add_argument('--num_classes',
                        type=int,
                        default=28,
                        help='Number of classes')
    parser.add_argument('--num_classes_iou',
                        type=int,
                        default=20,
                        help='Number of classes for iou calculation')
    parser.add_argument('--jobid')
    parser.add_argument('--experimentname')
    parser.add_argument('--dataroot')
    parser.add_argument('--folderroot',
                        default='/beegfs/work/breitenstein/')
    parser.add_argument('--imagenet-weight-path', help='path to the Imagenet pre-trained weights')#path to the pre-trained Imagenet weights

    #### Loss
    parser.add_argument('--loss', default="CE") #CE, MSE, MAE

# optimization
    parser.add_argument('--schedule', default="standard")
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=1)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)

    parser.add_argument('--weights_init', default='imagenet')
    parser.add_argument('--weight_type', default='erfnet')
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--iouVal', action='store_true', default=True)


    return parser

# activate determnistic flags
def deterministic(args):
#ä    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None, ):
        super().__init__()
        self.loss = torch.nn.NLLLoss(weight, reduction='none')

    def forward(self, outputs, targets):
        return torch.mean(self.loss(torch.nn.functional.log_softmax(outputs, dim=1), torch.squeeze(targets, dim=1)))


class AmodalCityscapes(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.files_in = sorted([f for f in os.listdir(root +'/'+ split +'/' + 'images/') if f.endswith('_leftImg8bit.png')])
        self.files_out = sorted([f for f in os.listdir(root +'/' + split + '/' + 'labels/' ) if f.endswith('.pt')])
    def __len__(self):
        return len(self.files_in)
    def __getitem__(self,idx):
        sample = ToTensor()(Image.open(os.path.join(self.root +'/' + self.split +'/'+ 'images/', self.files_in[idx])).convert('RGB'))
        label = torch.load(os.path.join(self.root +'/' + self.split +'/'+ 'labels/', self.files_out[idx]))
        return sample, label

class training():
    def __init__(self, args):
        self.args = args
        #self.savedir = f'../save/{self.opt.savedir}'

        slurmid = os.getenv('SLURM_JOB_ID',
                            datetime.now().strftime("%y-%m-%d_%H:%M:%S"))

        self.savedir = os.path.join('../save/',
                                         self.args.savedir)

        if not (self.args.non_deterministic):
            deterministic( self.args)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        print('We save into this directory: ', self.savedir)




        data_root = self.args.folderroot + self.args.dataroot

        print('dataroot for this training is', data_root)

        #load the Amodal Cityscapes Dataset
        train_dataset = AmodalCityscapes(data_root, 'train')
        val_dataset = AmodalCityscapes(data_root, 'val')

        #define the dataloader
        self.train_loader = DataLoader(train_dataset,
                            self.args.batch_size,
                            True,
                            num_workers=self.args.num_workers,
                            pin_memory=True,
                            drop_last=True)

        self.val_loader = DataLoader(val_dataset,
                                self.args.batch_size,
                                False,
                                num_workers=self.args.num_workers,
                                pin_memory=True,
                                drop_last=True)




        self.device = torch.device("cpu" if not self.opt.cuda else "cuda")


        if 'erfnet' in self.args.modeltype:
            spec = importlib.util.spec_from_file_location('erfnet', args.model + ".py")  # modelname, location
            model_file = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_file)
            if self.args.weights_init == 'imagenet':
                imagenet_encoder = torch.nn.DataParallel(1000)#imagenet size
                imagenet_encoder.load_state_dict(torch.load(self.args.imagenet_weight_path)['state_dict'])
                imagenet_encoder = next(imagenet_encoder.children()).features.encoder
                self.model = model_file.Net(self.args.num_classes,imagenet_encoder)
            else:
                self.model = model_file.Net(self.args.num_classes)
        else:
            raise ValueError("Only erfnet with imagenet initialization is implemented so far")



        self.model.to(self.device)
        self.model_optimizer = Adam(self.model.parameters(),
                                          self.args.lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.opt.num_epochs)),
                                    0.9)

        self.model_lr_scheduler = lr_scheduler.LambdaLR(
            self.model_optimizer, lr_lambda=lambda1)



        self.epoch = 0



        # create instances of classes used in the loss
        self.weight = torch.ones(self.args.num_classes)
        if 'erfnet' in self.args.weight_type:
            self.weight[0] = 6.9206132888794  # 2.3653597831726  #group 0
            self.weight[1] = 8.042702102410274  # 2.3653597831726  # group 1
            self.weight[2] = 9.673872550328584  # 2.3653597831726  #group 2
            self.weight[3] = 9.945302009582601  # 2.3653597831726  # group 3

            self.weight[4] = 0  # absence of group 0
            self.weight[5] = 0  # unlabeled (0)
            self.weight[6] = 2.8149201869965  # road (0)
            self.weight[7] = 6.9850029945374  # sidewalk (0)
            self.weight[8] = 3.7890393733978  # building (0)
            self.weight[9] = 9.9428062438965  # wall (0)
            self.weight[10] = 9.7702074050903  # fence (0)
            self.weight[11] = 9.5608062744141  # terrain (0)
            self.weight[12] = 7.8698215484619  # sky (0)
            self.weight[13] = 4.6323022842407  # vegetation (0)

            self.weight[14] = 2.3218942632177  # absence of group 1
            self.weight[15] = 9.5110931396484  # pole (1)
            self.weight[16] = 10.311357498169  # traffic light (1)
            self.weight[17] = 10.026463508606  # traffic sign (1)

            self.weight[18] = 2.3218942632177  # absence of group 2
            self.weight[19] = 6.6616044044495  # car (2)
            self.weight[20] = 10.260489463806  # truck (2)
            self.weight[21] = 10.287888526917  # bus (2)
            self.weight[22] = 10.289801597595  # train (2)
            self.weight[23] = 10.405355453491  # motorcycle (2)
            self.weight[24] = 10.138095855713  # bicycle (2)

            self.weight[25] = 2.3218942632177  # absence of group 3
            self.weight[26] = 9.5168733596802  # person (2)
            self.weight[27] = 10.373730659485  # rider (2)
        else:
            raise ValueError("Only erfnet with imagenet initialization is implemented so far")


        self.evaluator = evaluator(self.args.num_classes_iou,True)

        models_dir = os.path.join(self.savedir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.args.__dict__.copy()

        with open(os.path.join(models_dir, 'args.json'), 'w') as file:
            json.dump(to_save, file, indent=2)

        self.step = 0
        self.best_miou_on_validation = 0

        starttime = datetime.now().strftime("%H:%M")
        self.writer = SummaryWriter(
            '/beegfs/work/breitenstein/checkpoints/amodal_semantic_segmentation/4groups/' + '%s_%s_%s_%s' % (
            args.jobid, args.experimentname, date.today().strftime("%m-%d-%y"), starttime),
            comment='lr =%s, epochs=%s' % (self.args.lr, args.num_epochs))

        print('name of the SummaryWriter is',
              '%s_%s_%s_%s' % (args.jobid, args.experimentname, date.today().strftime("%m-%d-%y"), starttime))



    def train(self):
        """
        Run the entire training pipeline.

        :return:
        """

        start_time = time.time()
        for self.epoch in range(self.epoch, self.args.num_epochs):
            self.train_epoch()
            self.validation()
            print('Best mIoU on validation set: {}'.format(self.best_miou_on_validation),
              flush=True)
            if (self.epoch + 1) % self.args.epochs_save == 0:
                self.save_model()
        total_time = time.time() - start_time  # total time in seconds
        print("training time was ", total_time/(60*60), " hours.")



    def train_epoch(self):
        """
        Run a single epoch.

        :return:
        """

        print("Training epoch %d" %self.epoch, flush=True)
        epoch_loss = []
        epoch_loss_p = []
        epoch_loss_g0 = []
        epoch_loss_g1 = []
        epoch_loss_g2 = []
        epoch_loss_g3 = []
        self.model.train()
        for idx, (images, labels) in enumerate(self.train_loader):

            if self.args.cuda:
                images = images.cuda()
                labels[labels == 255] = 19
                labels = labels.permute(0, 3, 1, 2)
                labels.to(torch.device('cuda:0'))

            labels.to(torch.device('cuda:0'))

            #resize images and labels
            labels1 = torch.zeros((labels.shape[0], labels.shape[1], 512, 1024))
            images = F.interpolate(images, size=(512,1024),mode='bilinear')

            labels1[:,0,:,:] = F.interpolate(labels[:,0,:,:][:,np.newaxis,:,:], size=(512,1024),mode='nearest')[:,0,:,:]
            labels1[:,1,:,:] = F.interpolate(labels[:,1,:,:][:,np.newaxis,:,:], size=(512,1024),mode='nearest')[:,0,:,:]
            labels1[labels1 == -1] = 19

            labels = labels1.long().cuda()

            outputs_vis,outputs_inv, losses = self.apply_model(images, labels)

            epoch_loss.append(losses["loss"].item())
            epoch_loss_p.append(losses["loss_p"].item())
            epoch_loss_g0.append(losses["loss_g0"].item())
            epoch_loss_g1.append(losses["loss_g1"].item())
            epoch_loss_g2.append(losses["loss_g2"].item())
            epoch_loss_g3.append(losses["loss_g3"].item())

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()




            if idx%50==0 and idx>0:
                average_loss = sum(epoch_loss) / len(epoch_loss)
                print("Current loss for epoch: %d and step: %d is: %f" %(self.epoch, idx, average_loss))
                #uncomment for mIoU calculation on the training dataset
                # invis_labels = labels[:, 1, :, :].clone()
                # invis_labels = invis_labels[:, np.newaxis, :, :]
                # invis_labels[invis_labels == -1] = 19
                # vis_labels = labels[:, 0, :, :].clone()
                # vis_labels = vis_labels[:, np.newaxis, :, :]
                # self.evaluator.add_batch(vis_labels.long().data.cpu(), outputs_vis[None, None, :, :].data.cpu())
                # self.evaluator.add_batch_inv(invis_labels.long().data.cpu(),
                #                                  outputs_inv[None, None, :, :].data.cpu())
                # iou_visible = self.evaluator.mean_iou()
                # iou_invisible = self.evaluator.mean_iou_inv()
                # iou_total = self.evaluator.mean_iou_total()
                # print('Current step is', idx, 'with visible mIoU: ', iou_visible, ' invisible mIoU: ', iou_invisible,
                #       ' total mIoU: ', iou_total)


            self.step += 1
        self.model_lr_scheduler.step(self.epoch)
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

        self.writer.add_scalar('loss_p', sum(epoch_loss_p) / len(epoch_loss_p), self.epoch)
        self.writer.add_scalar('loss_g0', sum(epoch_loss_g0) / len(epoch_loss_g0), self.epoch)
        self.writer.add_scalar('loss_g1', sum(epoch_loss_g1) / len(epoch_loss_g1), self.epoch)
        self.writer.add_scalar('loss_g2', sum(epoch_loss_g2) / len(epoch_loss_g2), self.epoch)
        self.writer.add_scalar('loss_g3', sum(epoch_loss_g3) / len(epoch_loss_g3), self.epoch)
        self.writer.add_scalar('loss', average_epoch_loss_train, self.epoch)


    def validation(self):
        """
        Run the entire evaluation pipeline.

        :return:
        """
        self.evaluator_val = evaluator(self.args.num_classes_iou, invisible=True)
        self.model.eval()
        epoch_loss_val = []
        with torch.no_grad():
            for idx, (images, labels) in enumerate(self.val_loader):
                if self.args.cuda:
                    images = images.cuda()
                    labels[labels == 255] = 19
                    labels = labels.permute(0, 3, 1, 2)
                    labels.to(torch.device('cuda:0'))

                labels.to(torch.device('cuda:0'))

                # resize images and labels
                labels1 = torch.zeros((labels.shape[0], labels.shape[1], 512, 1024))
                images = F.interpolate(images, size=(512, 1024), mode='bilinear')

                labels1[:, 0, :, :] = F.interpolate(labels[:, 0, :, :][:, np.newaxis, :, :], size=(512, 1024),
                                                    mode='nearest')[:, 0, :, :]
                labels1[:, 1, :, :] = F.interpolate(labels[:, 1, :, :][:, np.newaxis, :, :], size=(512, 1024),
                                                    mode='nearest')[:, 0, :, :]
                labels1[labels1 == -1] = 19

                labels = labels1.long().cuda()

                outputs_vis, outputs_inv, losses = self.apply_model(images, labels)
                invis_labels = labels[:, 1, :, :].clone()
                invis_labels = invis_labels[:, np.newaxis, :, :]
                invis_labels[invis_labels == -1] = 19
                vis_labels = labels[:, 0, :, :].clone()
                vis_labels = vis_labels[:, np.newaxis, :, :]
                self.evaluator_val.add_batch(vis_labels.long().data.cpu(), outputs_vis[None, None, :, :].data.cpu())
                self.evaluator_val.add_batch_inv(invis_labels.long().data.cpu(), outputs_inv[None, None, :, :].data.cpu())
                epoch_loss_val.append(losses["loss"].item())

                if idx >0 and idx%50==0:
                    average = sum(epoch_loss_val) / len(epoch_loss_val)
                    print("Validation loss for epoch: %d in step: %idx is: %f" %(self.epoch, idx, average), flush=True)

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        self.writer.add_scalar('val_loss', average_epoch_loss_val, self.epoch)
        iou_visible = self.evaluator_val.mean_iou()
        iou_invisible = self.evaluator_val.mean_iou_inv()
        iou_total = self.evaluator_val.mean_iou_total()
        #meanacc = self.metric_model.pixel_accuracy()

        print("Visible mIoU on the validation dataset:", iou_visible, flush=True)
        print("Invisible mIoU on the validation dataset:", iou_invisible, flush=True)
        print("Total mIoU on the validation dataset:", iou_total, flush=True)

        if iou_visible > self.best_miou_on_validation:
            self.save_best_val_model()
            self.best_miou_on_validation = iou_visible

    def apply_model(self, images, labels):
        """
        Process a mini-batch.

        :param inputs: mini-batch (input + ground truth)
        :return: model output + loss values
        """


        outputs = self.model(images)
        losses = {}
        test_output = outputs[0, :, :, :].cpu().detach().numpy().transpose(1, 2,
                                                                           0)  # test_output.shape: (512, 1024, 26)
        test_output_p = outputs[0, :4, :, :].cpu().detach().numpy().transpose(1, 2,
                                                                              0)  # test_output_p.shape: (512, 1024, 3)
        test_output_g0 = outputs[0, 4:14, :, :].cpu().detach().numpy().transpose(1, 2, 0)
        test_output_g1 = outputs[0, 14:18, :, :].cpu().detach().numpy().transpose(1, 2, 0)
        test_output_g2 = outputs[0, 18:25, :, :].cpu().detach().numpy().transpose(1, 2, 0)
        test_output_g3 = outputs[0, 25:28, :, :].cpu().detach().numpy().transpose(1, 2, 0)
        inv_group = np.argsort(test_output_p.copy(), axis=-1)[:, :, -2]
        test_p = np.argmax(test_output_p, axis=-1)  # test_p.shape: (512, 1024)
        test_g0 = np.argmax(test_output_g0, axis=-1)
        test_g1 = np.argmax(test_output_g1, axis=-1)
        test_g2 = np.argmax(test_output_g2, axis=-1)
        test_g3 = np.argmax(test_output_g3, axis=-1)

        dict_g0 = {0: 19, 1: 19, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 10, 9: 8, 7: 9}
        dict_g1 = {0: 19, 1: 5, 2: 6, 3: 7}
        dict_g2 = {0: 19, 1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18}
        dict_g3 = {0: 19, 1: 11, 2: 12}

        ########## groups g0, g1, g2 #############
        for k, v in dict_g0.items(): test_g0[test_g0 == k] = v
        for k, v in dict_g1.items(): test_g1[test_g1 == k] = v
        for k, v in dict_g2.items(): test_g2[test_g2 == k] = v
        for k, v in dict_g3.items(): test_g3[test_g3 == k] = v

        ######## semantic segmentation ##############
        test_p[test_p == 3] = test_g3[test_p == 3]
        test_p[test_p == 2] = test_g2[test_p == 2]
        test_p[test_p == 1] = test_g1[test_p == 1]
        test_p[test_p == 0] = test_g0[test_p == 0]  # group 0 must be latest, because it includes 0, 1, 2
        # if prediction of semantic segmentation is absence of group1 or group2 or group3-> take background
        test_p[test_p == 19] = test_g0[test_p == 19]

        output_vis = torch.from_numpy(test_p)
        inv_group[inv_group == 3] = test_g3[inv_group == 3]
        inv_group[inv_group == 2] = test_g2[inv_group == 2]
        inv_group[inv_group == 1] = test_g1[inv_group == 1]
        inv_group[inv_group == 0] = test_g0[
            inv_group == 0]  # group 0 must be latest, because it includes 0, 1, 2
        # if prediction of semantic segmentation is absence of group1 or group2 -> take background
        inv_group[inv_group == 19] = test_g0[inv_group == 19]

        output_inv = torch.from_numpy(inv_group)

        output_g0 = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)  # output_g0.shape: (512, 1024, 3)
        output_g1 = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)
        output_g2 = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)
        output_g3 = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)
        output_ss = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)  # output_ss.shape: (512, 1024, 3)

        # convert trainId to color
        trainId2color = {label.trainId: label.color for label in reversed(labels_cityscape_seg.getlabels())}
        for k, v in trainId2color.items(): output_ss[test_p == k] = v
        for k, v in trainId2color.items(): output_g0[test_g0 == k] = v
        for k, v in trainId2color.items(): output_g1[test_g1 == k] = v
        for k, v in trainId2color.items(): output_g2[test_g2 == k] = v
        for k, v in trainId2color.items(): output_g3[test_g3 == k] = v

        target = labels.clone()  # .cpu()
        # print("target after labels.clone().cpu():", target.device)
        p = target.clone()
        mask_g0 = (p == 0) | (p == 1) | (p == 2) | (p == 3) | (p == 4) | (p == 8) | (p == 9) | (p == 10) | (p == 19)
        mask_g1 = (p == 5) | (p == 6) | (p == 7)
        mask_g2 = (p == 13) | (p == 14) | (p == 15) | (p == 16) | (p == 17) | (p == 18)
        mask_g3 = (p == 11) | (p == 12)
        p[mask_g0 == True] = 0
        p[mask_g1 == True] = 1
        p[mask_g2 == True] = 2
        p[mask_g3 == True] = 3

        compare_targets = torch.eq(p[:, 0, :, :], p[:, 1, :, :]).long()
        target[:, 1, :, :][compare_targets == 1] = 50
        target_vis = target[:, 0, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)
        target_occ = target[:, 1, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)

        # p = (p0, p1, p2),  q0 = (q00, q01, q02, q03, q04, q05, q06, q07, q08, q09), q1 = (q10, q11, q12, q13), q2 = (q20, q21, q22, q23, q24, q25, q26), q3 = (q30, q31, q32)

        ########################## calculating p0, p1, p2 ##########################
        target_p = p[:, 0, :, :]
        cuda0 = torch.device('cuda:0')
        ignore_only = torch.ones((self.args.height, self.args.width), device=cuda0) * 30
        torch_ones = torch.ones((self.args.batch_size, 1, self.args.height, self.args.width), device=cuda0)

        ########################## calculating q01-q09, q10-q13, q21-q28 ##########################
        group0 = [0, 1, 2, 3, 4, 8, 9, 10, 19]
        group1 = [5, 6, 7]
        group2 = [13, 14, 15, 16, 17, 18]
        group3 = [11, 12]

        q_ones = torch.ones(self.args.batch_size, 1, self.args.height, self.args.width)  # q05 - road, trainid(road) = 0
        # ==== group 0 ====
        # q01 is unlabeled, 19 is unlabeled
        q01_vis = torch.where(target_vis == 19, torch_ones * 1, ignore_only).long()
        q01_occ = torch.where(target_occ == 19, torch_ones * 1, ignore_only).long()
        # q02 road, 1 road is there, 30 is no road == > analogue with all train ids
        q02_vis = torch.where(target_vis == 0, torch_ones * 2, ignore_only).long()
        q02_occ = torch.where(target_occ == 0, torch_ones * 2, ignore_only).long()
        # q03 sidewalk, trainid = 1
        q03_vis = torch.where(target_vis == 1, torch_ones * 3, ignore_only).long()
        q03_occ = torch.where(target_occ == 1, torch_ones * 3, ignore_only).long()
        # q04 building, trainid = 2
        q04_vis = torch.where(target_vis == 2, torch_ones * 4, ignore_only).long()
        q04_occ = torch.where(target_occ == 2, torch_ones * 4, ignore_only).long()
        # q05 wall, trainid = 3
        q05_vis = torch.where(target_vis == 3, torch_ones * 5, ignore_only).long()
        q05_occ = torch.where(target_occ == 3, torch_ones * 5, ignore_only).long()
        # q06 fence, trainid = 4
        q06_vis = torch.where(target_vis == 4, torch_ones * 6, ignore_only).long()
        q06_occ = torch.where(target_occ == 4, torch_ones * 6, ignore_only).long()
        # q07 terrain, trainid = 9
        q07_vis = torch.where(target_vis == 9, torch_ones * 7, ignore_only).long()
        q07_occ = torch.where(target_occ == 9, torch_ones * 7, ignore_only).long()
        # q08 sky, trainid = 10
        q08_vis = torch.where(target_vis == 10, torch_ones * 8, ignore_only).long()
        q08_occ = torch.where(target_occ == 10, torch_ones * 8, ignore_only).long()
        # q09 vegetation, trainid = 8
        q09_vis = torch.where(target_vis == 8, torch_ones * 9, ignore_only).long()
        q09_occ = torch.where(target_occ == 8, torch_ones * 9, ignore_only).long()

        # ==== group 1 ====
        # q11 pole, trainid = 5
        q11_vis = torch.where(target_vis == 5, torch_ones * 1, ignore_only).long()
        q11_occ = torch.where(target_occ == 5, torch_ones * 1, ignore_only).long()
        # q12 traffic light, trainid = 6
        q12_vis = torch.where(target_vis == 6, torch_ones * 2, ignore_only).long()
        q12_occ = torch.where(target_occ == 6, torch_ones * 2, ignore_only).long()
        # q13 traffic sign, trainid = 7
        q13_vis = torch.where(target_vis == 7, torch_ones * 3, ignore_only).long()
        q13_occ = torch.where(target_occ == 7, torch_ones * 3, ignore_only).long()

        # ==== group 2 ====
        # q21 car, trainid = 13
        q21_vis = torch.where(target_vis == 13, torch_ones * 1, ignore_only).long()
        q21_occ = torch.where(target_occ == 13, torch_ones * 1, ignore_only).long()
        # q22 truck, trainid = 14
        q22_vis = torch.where(target_vis == 14, torch_ones * 2, ignore_only).long()
        q22_occ = torch.where(target_occ == 14, torch_ones * 2, ignore_only).long()
        # q23 bus, trainid = 15
        q23_vis = torch.where(target_vis == 15, torch_ones * 3, ignore_only).long()
        q23_occ = torch.where(target_occ == 15, torch_ones * 3, ignore_only).long()
        # q24 train, trainid = 16
        q24_vis = torch.where(target_vis == 16, torch_ones * 4, ignore_only).long()
        q24_occ = torch.where(target_occ == 16, torch_ones * 4, ignore_only).long()
        # q25motorcycle, trainid = 17
        q25_vis = torch.where(target_vis == 17, torch_ones * 5, ignore_only).long()
        q25_occ = torch.where(target_occ == 17, torch_ones * 5, ignore_only).long()
        # q26 bicycle, trainid = 18
        q26_vis = torch.where(target_vis == 18, torch_ones * 6, ignore_only).long()
        q26_occ = torch.where(target_occ == 18, torch_ones * 6, ignore_only).long()

        # ==== group 3 ====
        # q31 person, trainid = 11
        q31_vis = torch.where(target_vis == 11, torch_ones * 1, ignore_only).long()
        q31_occ = torch.where(target_occ == 11, torch_ones * 1, ignore_only).long()
        # q32 rider, trainid = 12
        q32_vis = torch.where(target_vis == 12, torch_ones * 2, ignore_only).long()
        q32_occ = torch.where(target_occ == 12, torch_ones * 2, ignore_only).long()

        ########################## calculating q00, q10, q20 ##########################
        # absence of the group = 1, group is there = 30
        # TRUE means group 1 is there, FALSE means group 1 is absent
        # TRUE has to map to 30, because group is not absent
        # FALSE has to map to 1 because group is absent
        q00_target_vis = p[:, 0, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)
        q00_target_occ = p[:, 1, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)

        q00_target_vis[q00_target_vis != 0] = 1
        q00_target_occ[q00_target_occ != 0] = 1
        q00 = (q00_target_vis & q00_target_occ).long()
        q00[q00 == 0] = 30
        q00[q00 == 1] = 0

        # absence of group g01
        q10_target_vis = p[:, 0, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)
        q10_target_occ = p[:, 1, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)
        q10_target_vis[q10_target_vis != 1] = 3
        q10_target_vis[q10_target_vis == 1] = 0
        q10_target_vis[q10_target_vis == 3] = 1
        q10_target_occ[q10_target_occ != 1] = 3
        q10_target_occ[q10_target_occ == 1] = 0
        q10_target_occ[q10_target_occ == 3] = 1

        q10 = (q10_target_vis & q10_target_occ).long()
        q10[q10 == 0] = 30
        q10[q10 == 1] = 0

        # absence of group g02
        q20_target_vis = p[:, 0, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)
        q20_target_occ = p[:, 1, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)

        q20_target_vis[q20_target_vis != 2] = 1
        q20_target_occ[q20_target_occ != 2] = 1
        q20_target_vis[q20_target_vis == 2] = 0
        q20_target_occ[q20_target_occ == 2] = 0
        q20 = (q20_target_vis & q20_target_occ).long()
        q20[q20 == 0] = 30
        q20[q20 == 1] = 0

        # absence of group g03
        q30_target_vis = p[:, 0, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)
        q30_target_occ = p[:, 1, :, :].clone().reshape(self.args.batch_size, 1, self.args.height, self.args.width)
        q30_target_vis[q30_target_vis != 3] = 1
        q30_target_occ[q30_target_occ != 3] = 1
        q30_target_vis[q30_target_vis == 3] = 0
        q30_target_occ[q30_target_occ == 3] = 0
        q30 = (q30_target_vis & q30_target_occ).long()
        q30[q30 == 0] = 30
        q30[q30 == 1] = 0

        ########################## losses p's ##########################
        loss_p = CrossEntropyLoss2d(self.weight[:4])(outputs[:, :4, :, :], target_p)
        # print("loss_p:", loss_p)
        ########################## losses q00, q10, q20 ##########################
        loss_q00 = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q00)
        loss_q10 = CrossEntropyLoss2d(self.weight[14:18])(outputs[:, 14:18, :, :], q10)
        loss_q20 = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q20)
        loss_q30 = CrossEntropyLoss2d(self.weight[25:28])(outputs[:, 25:28, :, :], q30)
        # print("loss_q00:", loss_q00)
        # print("loss_q10:", loss_q10)
        # print("loss_q20:", loss_q20)
        ########################## losses for visible ##########################
        # g0: q01-q09
        loss_q01_vis = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q01_vis)
        loss_q02_vis = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q02_vis)
        loss_q03_vis = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q03_vis)
        loss_q04_vis = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q04_vis)
        loss_q05_vis = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q05_vis)
        loss_q06_vis = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q06_vis)
        loss_q07_vis = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q07_vis)
        loss_q08_vis = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q08_vis)
        loss_q09_vis = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q09_vis)

        # g1: q11-q13
        loss_q11_vis = CrossEntropyLoss2d(self.weight[14:18])(outputs[:, 14:18, :, :], q11_vis)
        loss_q12_vis = CrossEntropyLoss2d(self.weight[14:18])(outputs[:, 14:18, :, :], q12_vis)
        loss_q13_vis = CrossEntropyLoss2d(self.weight[14:18])(outputs[:, 14:18, :, :], q13_vis)

        # g2: q21-q26
        loss_q21_vis = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q21_vis)
        loss_q22_vis = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q22_vis)
        loss_q23_vis = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q23_vis)
        loss_q24_vis = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q24_vis)
        loss_q25_vis = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q25_vis)
        loss_q26_vis = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q26_vis)

        # g3: q31-q32
        loss_q31_vis = CrossEntropyLoss2d(self.weight[25:28])(outputs[:, 25:28, :, :], q31_vis)
        loss_q32_vis = CrossEntropyLoss2d(self.weight[25:28])(outputs[:, 25:28, :, :], q32_vis)

        loss_vis_g0 = [loss_q01_vis, loss_q02_vis, loss_q03_vis, loss_q04_vis, loss_q05_vis, loss_q06_vis,
                       loss_q07_vis, loss_q08_vis, loss_q09_vis]
        loss_vis_g1 = [loss_q11_vis, loss_q12_vis, loss_q13_vis]
        loss_vis_g2 = [loss_q21_vis, loss_q22_vis, loss_q23_vis, loss_q24_vis, loss_q25_vis, loss_q26_vis]
        loss_vis_g3 = [loss_q31_vis, loss_q32_vis]

        ########################## losses for occluded ##########################
        # g0: q01-q09
        loss_q01_occ = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q01_occ)
        loss_q02_occ = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q02_occ)
        loss_q03_occ = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q03_occ)
        loss_q04_occ = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q04_occ)
        loss_q05_occ = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q05_occ)
        loss_q06_occ = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q06_occ)
        loss_q07_occ = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q07_occ)
        loss_q08_occ = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q08_occ)
        loss_q09_occ = CrossEntropyLoss2d(self.weight[4:14])(outputs[:, 4:14, :, :], q09_occ)

        # g1: q11-q13
        loss_q11_occ = CrossEntropyLoss2d(self.weight[14:18])(outputs[:, 14:18, :, :], q11_occ)
        loss_q12_occ = CrossEntropyLoss2d(self.weight[14:18])(outputs[:, 14:18, :, :], q12_occ)
        loss_q13_occ = CrossEntropyLoss2d(self.weight[14:18])(outputs[:, 14:18, :, :], q13_occ)

        # g2: q21-q28
        loss_q21_occ = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q21_occ)
        loss_q22_occ = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q22_occ)
        loss_q23_occ = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q23_occ)
        loss_q24_occ = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q24_occ)
        loss_q25_occ = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q25_occ)
        loss_q26_occ = CrossEntropyLoss2d(self.weight[18:25])(outputs[:, 18:25, :, :], q26_occ)

        # g3: q31-q32
        loss_q31_occ = CrossEntropyLoss2d(self.weight[25:28])(outputs[:, 25:28, :, :], q31_occ)
        loss_q32_occ = CrossEntropyLoss2d(self.weight[25:28])(outputs[:, 25:28, :, :], q32_occ)

        loss_occ_g0 = [loss_q01_occ, loss_q02_occ, loss_q03_occ, loss_q04_occ, loss_q05_occ, loss_q06_occ,
                       loss_q07_occ, loss_q08_occ, loss_q09_occ]
        loss_occ_g1 = [loss_q11_occ, loss_q12_occ, loss_q13_occ]
        loss_occ_g2 = [loss_q21_occ, loss_q22_occ, loss_q23_occ, loss_q24_occ, loss_q25_occ, loss_q26_occ]
        loss_occ_g3 = [loss_q31_occ, loss_q32_occ]

        l = 0.5

        loss_g0 = sum(loss_vis_g0) + l * sum(loss_occ_g0) + l * loss_q00
        loss_g1 = sum(loss_vis_g1) + l * sum(loss_occ_g1) + l * loss_q10
        loss_g2 = sum(loss_vis_g2) + l * sum(loss_occ_g2) + l * loss_q20
        loss_g3 = sum(loss_vis_g3) + l * sum(loss_occ_g3) + l * loss_q30

        loss = loss_p + loss_g0 + loss_g1 + loss_g2 + loss_g3

        losses["loss"] = loss
        losses["loss_p"] = loss_p
        losses["loss_g0"] = loss_g0
        losses["loss_g1"] = loss_g1
        losses["loss_g2"] = loss_g2
        losses["loss_g3"] = loss_g3
        return output_vis, output_inv, losses





    def save_best_val_model(self):
        """
        Best model saving function.

        :return:
        """


        save_path = os.path.join(self.savedir, "{}.pth".format("best_model_val"))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(self.savedir, "{}.pth".format("optim"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def save_model(self):
        """
        Model saving function.

        :return:
        """

        save_path = os.path.join(self.savedir, "{}_{}.pth".format("model",self.epoch))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(self.savedir, "{}_{}.pth".format("optimizer", self.epoch))
        torch.save(self.model_optimizer.state_dict(), save_path)







def main():
    """
    main function.

    :return:
    """
    args = get_paser_options()
    args = args.parse_args()

    trainer = training(args)
    trainer.train()


if __name__ == "__main__":
    main()
