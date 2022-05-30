import numpy as np
import torch
import os
import sys

sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/dataloader/')
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/dataloader/')
import importlib
import time
sys.path.append('/beegfs/work/breitenstein/Code/cv_repository/corner_case_detection/amodal_semantic_segmentation/Dataset_generation/')
from cp_utils import trainid2rgb
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pt_data_loader.specialdatasets import SimpleDataset
from definitions.labels_file import labels_cityscape_seg
import pt_data_loader.mytransforms as mytransforms
from evaluation import evaluator
import torch.nn.functional as F



NUM_CHANNELS = 3
NUM_CLASSES = 20#8
NUM_CLASSES_MIOU = 20
def load_my_state_dict(model, state_dict):  # load the models state dict
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    return model

class AmodalCityscapes(torch.utils.data.Dataset):#class to load the Amodal Cityscapes dataset
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.files_in = sorted([f for f in os.listdir(root + split +'/' + 'images/') if f.endswith('_leftImg8bit.png')])
        self.files_out = sorted([f for f in os.listdir(root + split + '/' + 'labels/' ) if f.endswith('.pt')])
    def __len__(self):
        return len(self.files_in)
    def __getitem__(self,idx):
        sample = ToTensor()(Image.open(os.path.join(self.root + self.split +'/'+ 'images/', self.files_in[idx])).convert('RGB'))
        label = torch.load(os.path.join(self.root + self.split +'/'+ 'labels/', self.files_out[idx]))
        return sample, label


def evaluate_iou(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath, flush=True)
    print ("Loading weights: " + weightspath, flush=True)


    spec = importlib.util.spec_from_file_location('erfnet', modelpath + ".py")  # modelname, location
    model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_file)

    model = model_file.Net(args.numclasses)#load the model

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # Load the weights and replace some names in the state dictionary to be compatible with the loaded model
    pretrained_dict = torch.load(weightspath)
    pretrained_dict = {k.replace('module.encoder.module.encoder.', 'module.encoder.'): v for k, v in pretrained_dict.items()}

    model = load_my_state_dict(model, pretrained_dict)#load the weights
    model.eval()

    labels_cs = labels_cityscape_seg.getlabels()
    train_ids = [labels_cs[i].trainId for i in range(len(labels_cs))]


    if 'amodal' in args.dataset:
        data_root = '/beegfs/data/shared/AmodalCityscapes/'
        test_dataset = AmodalCityscapes(data_root, 'test')
    else:#load the standard Cityscapes validation dataset using the IfN dataloader
        split = None
        labels_mode = 'fromid'
        height = 512
        width = 1024


        val_data_transforms = [mytransforms.CreateScaledImage(),
                               mytransforms.Resize((height, width)),
                               mytransforms.ConvertSegmentation(),
                               mytransforms.CreateColoraug(new_element=True),
                               mytransforms.RemoveOriginals(),
                               mytransforms.ToTensor(),
                               mytransforms.Relabel(255, 19)]

        test_dataset = SimpleDataset(dataset='cityscapes',
                                    trainvaltest_split='validation',
                                    video_mode='mono',
                                    stereo_mode='mono',
                                    labels=labels_cs,
                                    split=split,
                                    labels_mode=labels_mode,
                                    keys_to_load=['color', 'segmentation_trainid'],
                                    output_filenames=True,
                                    data_transforms=val_data_transforms)

    loader_test = DataLoader(test_dataset,
                            args.batch_size,
                            False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True)
#%%


    iou_evaluation = evaluator(20,True)#assuming we always want to calculate the miou inv and miou total
    #19 is the number of Cityscapes classes in this case

    start = time.time()


    for step, data in enumerate(loader_test):
        if 'amodal' in args.dataset:#if we use the amodal dataset
            images = data[0]
            labels = data[1]
            labels = labels.permute(0, 3, 1, 2)

            labels1 = torch.zeros((labels.shape[0], labels.shape[1], 512, 1024))
            images = F.interpolate(images, size=(512,1024),mode='bilinear')

            labels1[:,0,:,:] = F.interpolate(labels[:,0,:,:][:,np.newaxis,:,:], size=(512,1024),mode='nearest')[:,0,:,:]
            labels1[:,1,:,:] = F.interpolate(labels[:,1,:,:][:,np.newaxis,:,:], size=(512,1024),mode='nearest')[:,0,:,:]

            labels = labels1.long().cuda()
        else: #use the standard cityscapes dataset
            images = data["color_aug", 0, 0]
            labels = data['segmentation_trainid', 0, 0].long()

        images = images.cuda()
        labels = labels.long().cuda()
        labels[labels == 255] = 19

        labels.to(torch.device('cuda:0'))

        inputs = Variable(images)

        with torch.no_grad():


            if 'amodal' in args.modeltype:#if the model predicts both visible and amodal labels
                outputs = model(inputs)  # get the outputs of the model
                softmax_vals = torch.nn.Softmax(dim=1)(outputs) #apply softmax to the outputs

                if args.numgroups ==4:#if K=4 groups, shape of softmax vals batchsize x 28 x h x w
                    test_output_p = softmax_vals[0, :4, :, :].cpu().detach().numpy().transpose(1, 2, 0)
                    test_output_g0 = softmax_vals[0, 4:14, :, :].cpu().detach().numpy().transpose(1, 2, 0)
                    test_output_g1 = softmax_vals[0, 14:18, :, :].cpu().detach().numpy().transpose(1, 2, 0)
                    test_output_g2 = softmax_vals[0, 18:25, :, :].cpu().detach().numpy().transpose(1, 2, 0)
                    test_output_g3 = softmax_vals[0, 25:28, :, :].cpu().detach().numpy().transpose(1, 2, 0)

                    inv_group = np.argsort(test_output_p.copy(), axis=-1)[:, :, -2]
                    test_p = np.argmax(test_output_p, axis=-1) #argmax of p vector is visible group
                    test_g0 = np.argmax(test_output_g0, axis=-1) #argmax is visible class of group 0
                    test_g1 = np.argmax(test_output_g1, axis=-1) #argmax is visible class of group 1
                    test_g2 = np.argmax(test_output_g2, axis=-1) #argmax is visible class of group 2
                    test_g3 = np.argmax(test_output_g3, axis=-1) #argmax is visible class of group 3

                    dict_g0 = {0: 19, 1: 19, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 10, 9: 8, 7: 9}
                    dict_g1 = {0: 19, 1: 5, 2: 6, 3: 7}
                    dict_g2 = {0: 19, 1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18}
                    dict_g3 = {0: 19, 1: 11, 2: 12}

                    ########## groups g0, g1, g2, g3 #############
                    for k, v in dict_g0.items(): test_g0[test_g0 == k] = v
                    for k, v in dict_g1.items(): test_g1[test_g1 == k] = v
                    for k, v in dict_g2.items(): test_g2[test_g2 == k] = v
                    for k, v in dict_g3.items(): test_g3[test_g3 == k] = v

                    ######## semantic segmentation ##############
                    test_p[test_p == 3] = test_g3[test_p == 3]
                    test_p[test_p == 2] = test_g2[test_p == 2]
                    test_p[test_p == 1] = test_g1[test_p == 1]
                    test_p[test_p == 0] = test_g0[test_p == 0]  # group 0 must be latest, because it includes 0, 1, 2
                    # if prediction of semantic segmentation is absence of group1 or group2 -> take background
                    test_p[test_p == 19] = test_g0[test_p == 19]

                    ######## amodal semantic segmentation ##############
                    inv_group[inv_group == 3] = test_g3[inv_group == 3]
                    inv_group[inv_group == 2] = test_g2[inv_group == 2]
                    inv_group[inv_group == 1] = test_g1[inv_group == 1]
                    inv_group[inv_group == 0] = test_g0[
                        inv_group == 0]  # group 0 must be latest, because it includes 0, 1, 2
                    # if prediction of semantic segmentation is absence of group1 or group2 -> take background
                    inv_group[inv_group == 19] = test_g0[inv_group == 19]

                    output_inv = torch.from_numpy(inv_group)
                    inv_group[inv_group == 19] = -1
                    test_g0[test_g0 == 19] = -1
                    test_g2[test_g2 == 19] = -1
                    test_g1[test_g1 == 19] = -1
                    test_g3[test_g3 == 19] = -1
                    output_ss = torch.from_numpy(test_p) # predicted visible labels

                    if 'amodal' in args.dataset:
                        invis_labels = labels[:, 1, :, :]
                        invis_labels = invis_labels[:, np.newaxis, :, :]
                        invis_labels[invis_labels == -1] = 19
                    else:
                        invis_labels = labels[:, 0, :, :]
                        invis_labels = invis_labels[:, np.newaxis, :, :]
                        invis_labels[invis_labels == -1] = 19
                    vis_labels = labels[:, 0, :, :]
                    vis_labels = vis_labels[:, np.newaxis, :, :]
                    if step == 0:
                        torch.save(output_ss[None, None, :, :].data.cpu(), 'outputs_network_k4.pt')
                        torch.save(output_inv[None, None, :, :].data.cpu(), 'outputs_inv_network_k4.pt')
                        torch.save(invis_labels.long().data.cpu(), 'invis_labels_k4.pt')
                        torch.save(vis_labels.long().data.cpu(), 'vis_labels_k4.pt')
                    iou_evaluation.add_batch(vis_labels.long().data.cpu(),output_ss[None, None, :, :].data.cpu())
                    iou_evaluation.add_batch_inv(invis_labels.long().data.cpu(),output_inv[None,None,:,:].data.cpu())

                if args.numgroups == 3:#if K=3 groups, shape of softmax vals batchsize x 26 x h x w
                    test_output_p = softmax_vals[0, :3, :, :].cpu().detach().numpy().transpose(1, 2,
                                                                                               0)  # test_output_p.shape: (512, 1024, 3)
                    test_output_g0 = softmax_vals[0, 3:13, :, :].cpu().detach().numpy().transpose(1, 2, 0)
                    test_output_g1 = softmax_vals[0, 13:17, :, :].cpu().detach().numpy().transpose(1, 2, 0)
                    test_output_g2 = softmax_vals[0, 17:26, :, :].cpu().detach().numpy().transpose(1, 2, 0)

                    inv_group = np.argsort(test_output_p, axis=-1)[:, :, -2]
                    test_p = np.argmax(test_output_p, axis=-1)  # test_p.shape: (512, 1024)
                    test_g0 = np.argmax(test_output_g0, axis=-1)
                    test_g1 = np.argmax(test_output_g1, axis=-1)
                    test_g2 = np.argmax(test_output_g2, axis=-1)
                    dict_g0 = {0: 19, 1: 19, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 10, 9: 8, 7: 9}
                    dict_g1 = {0: 19, 1: 5, 2: 6, 3: 7}
                    dict_g2 = {0: 19, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18}

                    ########## groups g0, g1, g2 #############
                    for k, v in dict_g0.items(): test_g0[test_g0 == k] = v
                    for k, v in dict_g1.items(): test_g1[test_g1 == k] = v
                    for k, v in dict_g2.items(): test_g2[test_g2 == k] = v

                    ######## semantic segmentation ##############
                    test_p[test_p == 2] = test_g2[test_p == 2]
                    test_p[test_p == 1] = test_g1[test_p == 1]
                    test_p[test_p == 0] = test_g0[test_p == 0]  # group 0 must be latest, because it includes 0, 1, 2
                    # if prediction of semantic segmentation is absence of group1 or group2 -> take background
                    test_p[test_p == 19] = test_g0[test_p == 19]

                    output_ss = torch.from_numpy(test_p)  # torch.Size([512, 1024])

                    ######## amodal semantic segmentation ##############
                    inv_group[inv_group == 2] = test_g2[inv_group == 2]
                    inv_group[inv_group == 1] = test_g1[inv_group == 1]
                    inv_group[inv_group == 0] = test_g0[inv_group == 0]
                    output_inv = torch.from_numpy(inv_group)
                    #inv_group[inv_group == 19] = -1
                    test_g0[test_g0 == 19] = -1
                    test_g2[test_g2 == 19] = -1
                    test_g1[test_g1 == 19] = -1
                    #test_g3[test_g3 == 19] = -1
                    output_ss = torch.from_numpy(test_p)  # predicted visible labels

                    if 'amodal' in args.dataset:
                        invis_labels = labels[:, 1, :, :]
                        invis_labels = invis_labels[:, np.newaxis, :, :]
                        invis_labels[invis_labels == -1] = 19
                    else:
                        invis_labels = labels[:, 0, :, :]
                        invis_labels = invis_labels[:, np.newaxis, :, :]
                        invis_labels[invis_labels == -1] = 19
                    vis_labels = labels[:, 0, :, :]
                    vis_labels = vis_labels[:, np.newaxis, :, :]


                    # output_inv[output_inv == -1] = 19

                    #add both the visible and invisible predictions to our evaluator
                    iou_evaluation.add_batch(vis_labels.long().data.cpu(),output_ss[None, None, :, :].data.cpu())
                    iou_evaluation.add_batch_inv(invis_labels.long().data.cpu(),output_inv[None,None,:,:].data.cpu())
            else: #if we use the normal model
                if 'amodal' in args.dataset:
                    vis_labels = labels[:, 0, :, :]
                    vis_labels = vis_labels[:, np.newaxis, :, :]
                    invis_labels = labels[:, 1, :, :]
                    invis_labels = invis_labels[:, np.newaxis, :, :]
                else: #for the standard dataset we set visible and invisible labels equal
                    # however we recommend not to consider the miou invisible and miou total in this case
                    # because their calculation does not make sense, especially miou = miou_inv
                    vis_labels = labels[:, 0, :, :]
                    vis_labels = vis_labels[:, np.newaxis, :, :]
                    invis_labels = labels[:, 0, :, :]
                    invis_labels = invis_labels[:, np.newaxis, :, :]
                invis_labels[invis_labels == -1] = 19



                outputs = model(inputs)
                softmax_vals = torch.nn.Softmax(dim=1)(outputs)
                # add both the visible and invisible predictions to our evaluator
                comb_vis = 0.5*images[0].permute(1,2,0).cpu().numpy()+ 0.5*trainid2rgb(softmax_vals.max(1)[1].unsqueeze(1).data.cpu()[0,0]).numpy()

                if args.plot:
                    plt.imsave(
                        args.plotpath + 'visible_pred_labels_%d.png' % step,comb_vis)
                iou_evaluation.add_batch(vis_labels.long().data.cpu(), softmax_vals.max(1)[1].unsqueeze(1).data.cpu())
                iou_evaluation.add_batch_inv(invis_labels.long().data.cpu(), softmax_vals.max(1)[1].unsqueeze(1).data.cpu())


        if (step + 1) % 100 == 0:
            print('{}/500'.format(step+1), flush=True)

    iou_visible = iou_evaluation.mean_iou()
    iou_invisible = iou_evaluation.mean_iou_inv()
    iou_total = iou_evaluation.mean_iou_total()




    print("Evaluation time: ", time.time()-start)
    print("Results on the", args.dataset, "dataset are:")
    print("Visible mIoU: ", iou_visible)
    print("Invisible mIoU: ", iou_invisible)
    print("Total mIoU:", iou_total)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--numgroups', type=int, default=4)
    parser.add_argument('--numclasses', type=int, default=28)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dataset', default = 'amodal') #amodal or standard meaning which dataset to evaluate on
    parser.add_argument('--modeltype', default='amodal') #amodal or standard
    parser.add_argument('--plot', default=False)
    parser.add_argument('--plotpath', default='fusion16/CSVal/')

    evaluate_iou(parser.parse_args())
