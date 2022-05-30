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
                        default=26,
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
            self.weight[0] = 2.8149201869965
            self.weight[1] = 6.9850029945374
            self.weight[2] = 3.7890393733978
            self.weight[3] = 9.9428062438965
            self.weight[4] = 9.7702074050903
            self.weight[5] = 9.5110931396484
            self.weight[6] = 10.311357498169
            self.weight[7] = 10.026463508606
            self.weight[8] = 4.6323022842407
            self.weight[9] = 9.5608062744141
            self.weight[10] = 7.8698215484619
            self.weight[11] = 9.5168733596802
            self.weight[12] = 10.373730659485
            self.weight[13] = 6.6616044044495
            self.weight[14] = 10.260489463806
            self.weight[15] = 10.287888526917
            self.weight[16] = 10.289801597595
            self.weight[17] = 10.405355453491
            self.weight[18] = 10.138095855713
        else:
            raise ValueError("Only erfnet with imagenet initialization is implemented so far")

        self.loss_func = CrossEntropyLoss2d(self.weight)
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

            outputs_vis, losses = self.apply_model(images, labels)

            epoch_loss.append(losses["loss"].item())


            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()




            if idx%50==0 and idx>0:
                average_loss = sum(epoch_loss) / len(epoch_loss)
                print("Current loss for epoch: %d and step: %d is: %f" %(self.epoch, idx, average_loss))



            self.step += 1
        self.model_lr_scheduler.step(self.epoch)
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

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

                outputs_vis, losses = self.apply_model(images, labels)

                vis_labels = labels[:, 0, :, :].clone()
                vis_labels = vis_labels[:, np.newaxis, :, :]
                self.evaluator_val.add_batch(vis_labels.long().data.cpu(), outputs_vis[None, None, :, :].data.cpu())
                epoch_loss_val.append(losses["loss"].item())

                if idx >0 and idx%50==0:
                    average = sum(epoch_loss_val) / len(epoch_loss_val)
                    print("Validation loss for epoch: %d in step: %idx is: %f" %(self.epoch, idx, average), flush=True)

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        self.writer.add_scalar('val_loss', average_epoch_loss_val, self.epoch)
        iou_visible = self.evaluator_val.mean_iou()
        iou_total = self.evaluator_val.mean_iou_total()
        #meanacc = self.metric_model.pixel_accuracy()

        print("Visible mIoU on the validation dataset:", iou_visible, flush=True)
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



        loss = self.loss_func(outputs, targets)

        losses["loss"] = loss
        return outputs, losses





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
