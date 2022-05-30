import torch


class evaluator(object):

    def __init__(self, num_classes, invisible=False):
        self.num_classes = num_classes
        """
        confusion matrix is of size num_classes * num_classes
        notation: actual class (predicted class) --> FN for actual class, FP for predicted class, or TP if both equal
                         |  predicted class      |
                         |  0    |   1   |  2   | ...
        true       |  0  |  0(0) | 0(1)  | 0(2) |
        class      |  1  |  1(0) |  1(1) | 1(2) |
                   |  2  |  2(0) |  2(1) | 2(2) |
                   | ... |       
        """
        self.confusion_matrix = torch.zeros(self.num_classes,self.num_classes)
        self.invisible = invisible
        if self.invisible:#if we want to calculate also invisible ious or joint ious we need to consider also invisible labels
            self.confusion_matrix_invisible = torch.zeros(self.num_classes,self.num_classes)

    def iou_per_class(self):
        """

        :return: iou per class for the given data
        """
        conf_matrix = self.confusion_matrix[0:-1,0:-1]
        tp = torch.diag(conf_matrix)
        fp = torch.sum(conf_matrix,dim=0) -tp
        fn = torch.sum(conf_matrix, dim=1) -tp
        denominator = fp+fn #sum includes twice tp so for iou denominator we need to substract tp once
        iou = tp/denominator
        return iou

    def mean_iou(self):
        """

        :return: miou calculated for the given data
        """
        conf_matrix = self.confusion_matrix[0:-1, 0:-1]
        tp = torch.diag(conf_matrix)
        fp_fn = torch.sum(conf_matrix,dim=0) + torch.sum(conf_matrix, dim=1)
        denominator = fp_fn - tp #sum includes twice tp so for iou denominator we need to substract tp once
        iou = tp/denominator
        miou = torch.nanmean(iou)
        return miou


    def iou_inv_per_class(self):
        """

        :return: iou per class for the given data
        """
        assert self.invisible, f"we can only calculate the iou_inv if we have amodal labels: {self.invisible}"
        conf_matrix = self.confusion_matrix_invisible[0:-1, 0:-1]
        tp = torch.diag(conf_matrix)
        fp_fn = torch.sum(conf_matrix,dim=0) + torch.sum(conf_matrix, dim=1)
        denominator = fp_fn - tp #sum includes twice tp so for iou denominator we need to substract tp once
        iou = tp/denominator
        return iou

    def mean_iou_inv(self):
        """

        :return: miou calculated for the given data
        """

        assert self.invisible, f"we can only calculate the iou_inv if we have amodal labels: {self.invisible}"
        conf_matrix = self.confusion_matrix_invisible[0:-1, 0:-1]
        tp = torch.diag(conf_matrix)
        fp_fn = torch.sum(conf_matrix, dim=0) + torch.sum(conf_matrix, dim=1)
        denominator = fp_fn - tp #sum includes twice tp so for iou denominator we need to substract tp once
        iou = tp/denominator
        miou = torch.nanmean(iou)
        return miou

    def iou_total_per_class(self):
        """

        :return: iou total per class for the given data
        """
        assert self.invisible, f"we can only calculate the iou_total if we have amodal labels: {self.invisible}"
        conf_matrix_invisible = self.confusion_matrix_invisible[0:-1, 0:-1]
        tp_inv = torch.diag(conf_matrix_invisible)
        fp_fn_inv = torch.sum(conf_matrix_invisible,dim=0) + torch.sum(conf_matrix_invisible, dim=1)
        conf_matrix = self.confusion_matrix[0:-1, 0:-1]
        tp = torch.diag(conf_matrix)
        fp_fn = torch.sum(conf_matrix, dim=0) + torch.sum(conf_matrix, dim=1)

        denominator_vis = fp_fn - tp #sum includes twice tp so for iou denominator we need to substract tp once
        denominator_inv = fp_fn_inv - tp_inv #sum includes twice tp so for iou denominator we need to substract tp once

        iou = (tp+tp_inv)/(denominator_vis+denominator_inv)
        return iou

    def mean_iou_total(self):
        """

        :return: miou total calculated for the given data
        """

        assert self.invisible, f"we can only calculate the iou_total if we have amodal labels: {self.invisible}"
        conf_matrix_invisible = self.confusion_matrix_invisible[0:-1, 0:-1]
        tp_inv = torch.diag(conf_matrix_invisible)
        fp_fn_inv = torch.sum(conf_matrix_invisible, dim=0) + torch.sum(conf_matrix_invisible, dim=1)
        conf_matrix = self.confusion_matrix[0:-1, 0:-1]
        tp = torch.diag(conf_matrix)
        fp_fn = torch.sum(conf_matrix, dim=0) + torch.sum(conf_matrix, dim=1)
        denominator_vis = fp_fn - tp #sum includes twice tp so for iou denominator we need to substract tp once
        denominator_inv = fp_fn_inv - tp_inv #sum includes twice tp so for iou denominator we need to substract tp once

        iou = (tp+tp_inv)/(denominator_vis+denominator_inv)
        miou = torch.nanmean(iou)
        return miou

    def add_batch(self,gt,pred):
        """
        update the confusion matrix with the current batch
        :param gt: ground truth sem seg mask
        :param pred: predicted sem seg mask
        :return: none
        """

        #filter out ignore classes (assuming we ignore num_classes+1)
        valid_classes = (gt >= 0) & (gt < self.num_classes)
        double_labels = self.num_classes*gt[valid_classes].int() + pred[valid_classes]
        counts = torch.bincount(double_labels,minlength=self.num_classes*self.num_classes)

        self.confusion_matrix+= counts.reshape(self.num_classes,self.num_classes)

    def add_batch_inv(self,gt,pred):
        """
        update the invisible confusion matrix with the current batch
        :param gt: ground truth sem seg mask
        :param pred: predicted sem seg mask
        :return: none
        """

        #filter out ignore classes (assuming we ignore num_classes+1)
        valid_classes = (gt >= 0) & (gt < self.num_classes)

        double_labels = self.num_classes*gt[valid_classes].int() + pred[valid_classes]

        counts = torch.bincount(double_labels,minlength=self.num_classes*self.num_classes)

        self.confusion_matrix_invisible+= counts.reshape(self.num_classes,self.num_classes)

    def add_batch_joint(self,gt,pred):
        """
        use this method for visible and invisible confusion matrices
        :param gt: ground truth sem seg mask
        :param pred: predicted sem seg mask
        :return:  none
        """
        assert self.invisible, f"we can only add joint batches if we also have invisible labels: {self.invisible}"

        gt_visible = gt[:, 0, :, :][:, None, :, :]
        gt_invisible = gt[:, 1, :, :][:, None, :, :]
        pred_visible = pred[:, 0, :, :][:, None, :, :]
        pred_invisible = pred[:, 1, :, :][:, None, :, :]
        valid_classes = (gt >= 0) & (gt < self.num_classes)
        double_labels_visible = self.num_classes*gt_visible[valid_classes].int() + pred_visible[valid_classes]
        counts_visible = torch.bincount(double_labels_visible,minlength=self.num_classes*self.num_classes)
        self.confusion_matrix+= counts_visible.reshape(self.num_classes,self.num_classes)

        valid_classes = (gt >= 0) & (gt < self.num_classes)
        double_labels_invisible = self.num_classes*gt_invisible[valid_classes].astype('int') + pred_invisible[valid_classes]
        counts_invisible = torch.bincount(double_labels_invisible,minlength=self.num_classes*self.num_classes)
        self.confusion_matrix_invisible+= counts_invisible.reshape(self.num_classes,self.num_classes)